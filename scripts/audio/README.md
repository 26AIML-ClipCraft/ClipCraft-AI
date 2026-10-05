# ClipCraft 오디오-비주얼 ReVisionLLM 실행 가이드 (단일 노드, GPU 2장 병렬 추출 / 학습은 GPU 1장)

실험 설계 문서: https://claude.ai/code/artifact/69f2dbb4-669a-4a59-a299-093bf1a495a5

이 문서는 환경 구축부터 최종 ablation 리포트까지 **실제로 명령을 치는 순서**를 적은 것이다.
모든 스크립트는 저장소 루트(`ReVisionLLM/`)에서 실행한다. 한 번에 최대 1일만 돌릴 수 있으므로
학습·추출 스크립트는 20시간에 스스로 멈추고, **같은 명령을 다시 실행하면 이어서 진행**된다.

> **현재 목표: 1차 비교.** R0(공개 체크포인트), R1(visual-only, Stage B까지), R2(audio-visual, Stage A→B까지)를
> Test-sub에서 비교한다. Stage C(7-5), 시드 3개, 보조 행(R3, R5~R7)은 오디오 이득이 확인된 뒤의 후속 단계다.
> 1차 비교에는 1~7-3(Stage B까지), 6, 8의 R0·R1·R2·R4, 9단계가 필요하다. 자세한 내용은 루트의 `PLAN.md`.

> 주의: 이 코드는 작성만 되어 있고 아직 한 번도 실행되지 않았다. 0단계와 6단계(R0 재현)에서
> 오류가 나면 거기서 먼저 고친 뒤 학습으로 넘어갈 것.

---

## 0. 환경 구축 (약 반나절)

```bash
conda create -n revaudio python=3.10 -y
conda activate revaudio

# 4090(Ada, sm_89)과 3090(Ampere, sm_86) 모두 cu121 wheel을 쓴다. 학습 시간 추정치는 4090 기준이며 3090(gpu4)은 더 걸릴 수 있다.
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 시스템 ffmpeg (CLIP 프레임 디코딩, CLAP 오디오 디코딩, yt-dlp 병합에 모두 필요)
conda install -c conda-forge ffmpeg -y        # 또는 apt install ffmpeg

python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

DeepSpeed와 flash-attn은 설치하지 않는다. 학습은 HF Trainer + SDPA attention으로 단일 GPU에서 돈다.

---

## 1. 가중치 다운로드 (약 20GB)

`checkpoints/` 아래에 다음 구조로 둔다. 경로를 바꾸면 `scripts/audio/env.sh`의 변수를 함께 바꾼다.

| 변수 (env.sh) | 경로 | 출처 |
| --- | --- | --- |
| `VICUNA` | `checkpoints/vicuna-7b-v1.5/` | HuggingFace `lmsys/vicuna-7b-v1.5`. 이 저장소는 safetensors 없이 `pytorch_model-*.bin`(약 13.5GB)만 제공한다 |
| `CLIP_PT` | `checkpoints/clip/ViT-L-14.pt` | OpenAI CLIP |
| `VTIMELLM_STAGE1_PROJ` | `checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin` | VTimeLLM stage1 (README의 VTimeLLM 링크) |
| `PUBLIC_STAGE1_SPARSE` | `checkpoints/public/chapters_stage1_sparse/non_lora_trainables.bin` | ReVisionLLM 공개 가중치 (Google Drive, README 링크) |
| `PUBLIC_STAGE2` | `checkpoints/public/chapters_stage2_long_100/` | 위 Drive의 chapters stage2. `adapter_config.json`, `adapter_model.*`, `non_lora_trainables.bin`이 있어야 한다 |

```bash
pip install "huggingface_hub[cli]==0.24.7"   # -U 금지: 1.x 이상은 transformers 4.41.2와 충돌하고 huggingface-cli도 없다
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir checkpoints/vicuna-7b-v1.5 \
    --include "*.json" "*.model" "*.bin"
mkdir -p checkpoints/clip && wget -O checkpoints/clip/ViT-L-14.pt \
    https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
# VTimeLLM / ReVisionLLM 가중치는 브라우저로 받아 위 경로에 배치
#  - ReVisionLLM 공개 가중치: Google Drive (저장소 README의 'trained models weights' 링크)
#  - VTimeLLM: Tsinghua Cloud (https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/). stage1~3이 묶인 tar로 오므로
#    stage1 projector만 꺼낸다:
#      tar -xf vtimellm-vicuna-v1-5-7b.tar --strip-components=1 vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
#          -C checkpoints/vtimellm-vicuna-v1-5-7b-stage1/
```

CLAP(`laion/clap-htsat-fused`)은 첫 추출 시 transformers가 자동으로 받는다.

---

## 2. VidChapters-7M 메타데이터 준비

VidChapters 저장소에서 VMR(video moment retrieval) 분할 파일을 받아 `data/chapters/`에 둔다.

```
data/chapters/chapters_vmr_train.jsonl
data/chapters/chapters_vmr_val.jsonl
data/chapters/chapters_vmr_test.jsonl
```

train은 한 줄이 영상 하나(`vid`=11자 YouTube id, `query`/`relevant_windows`는 리스트)이고,
val/test는 한 줄이 query 하나다. val/test의 `vid`는 `"<챕터 번호><YouTube id>"`(예: `3uwN1yCPgJgk`)라서
뒤 11자가 실제 영상 id다. 스크립트는 모두 이 11자 id 기준으로 영상 단위로 묶어 쓴다.
원본 영상은 여기서 받지 않는다. 3단계에서 서브셋만 내려받는다.

---

## 3. 서브셋 선정 → 다운로드 → feature 추출 (Train-sub 2,500시간 + Val/Test, GPU 2장 병렬. 소요 시간은 실측 후 갱신)

```bash
# 3-1. 공식 split에서 후보 샘플링 (seed 42). Train-sub는 기본 2,500시간 (TRAIN_HOURS=2500).
#      1차 비교(짧은 학습)는 TRAIN_HOURS=1600 (약 1만 영상, query 약 6.8만 개)으로 충분하다. 시간 수가 달라도 작은 쪽이
#      큰 쪽의 부분집합이라(층별 앞부분) 나중에 TRAIN_HOURS를 키워 다시 실행하면 이어서 늘어난다.
bash scripts/audio/data_pipeline.sh subsets
#   -> data/chapters_audio/subsets/{train_sub,val_sub,test_sub}_candidates.json
#   val/test의 vid는 "<챕터번호><YouTube id>"라서 뒤 11자로 영상 단위로 묶어 샘플링한다.

# 3-2. 병렬 추출 (권장): GPU 2장, GPU당 워커 4개 = 워커 8개 (gpu4 3090 실측: 워커 6개 약 92배, 8개 약 119배 실시간. GPU당 VRAM 약 17GB). 워커마다 shard LMDB에 쓰고 끝나면 메인 LMDB로 병합한다.
#      Test-sub는 오디오 트랙(m4a)을 data/chapters_audio/test_audio_keep/ 에 백업한다.
#      val/test는 n_final(Val 300 / Test 1,000) + 10% 예비만 받고, 실패한 영상은 다음 후보로 대체된다.
GPUS="0 1" PROCS_PER_GPU=4 bash scripts/audio/data_pipeline.sh extract_par test
GPUS="0 1" PROCS_PER_GPU=4 bash scripts/audio/data_pipeline.sh extract_par val
# 3-3. Train-sub. 20시간에 멈추면 같은 명령을 다시 실행한다(이미 저장된 id는 건너뜀).
GPUS="0 1" PROCS_PER_GPU=4 bash scripts/audio/data_pipeline.sh extract_par train
#   튜닝: PROCS_PER_GPU(VRAM 약 4GB/워커), DL_WORKERS(워커당 다운로드 스레드, 기본 2), FFMPEG_THREADS(기본 3),
#         OMP_THREADS(기본 2), CLIP_BATCH(기본 128). 진행: subsets/extract_<split>_w*.jsonl, 워커 로그: *.out
#   Train은 후보 목록이 영상 길이 순으로 정렬되어 있어서 연속 구간으로 나누면 한 워커가 몇 시간짜리 영상을 전부 맡는다(실측: 워커 간 109h vs 1,104h).
#   extract_par train은 목록을 섞어(seed 고정) 워커에 번갈아 나눈다(--interleave). 그래서 워커 부하가 균등하고,
#   중간에 멈춰도 저장된 영상이 Train-sub의 균일한 무작위 부분집합이다(멈춘 경우 `data_pipeline.sh merge train` 후 prep).
#   (단일 프로세스 방식은 `extract test|val|train [i N]`; val/test는 --target n_final에서 멈춤)

# 3-4. 나중에 Train-sub를 10,000시간으로 늘리기: TRAIN_HOURS=10000 으로 subsets를 다시 실행한 뒤
#      extract_par train 을 다시 실행하면 된다. 2,500시간 샘플은 10,000시간 샘플의 부분집합이고(같은 seed),
#      val/test 후보는 그대로이며, 이미 저장된 영상은 건너뛰므로 새로 필요한 영상만 받는다.
#      (학습은 늘어난 데이터로 Stage B부터 다시 돌려야 한다. 2,500시간 결과와는 다른 run이다.)
```

결과는 두 개의 LMDB이다.

| 경로 | 내용 | 용량 (2,500시간 기준 / 1만 시간 기준) |
| --- | --- | --- |
| `data/chapters_audio/clip_l14_2fps_lmdb` | CLIP ViT-L/14, 2fps, fp16 | 약 28GB / 약 110GB |
| `data/chapters_audio/clap_1s_lmdb` | CLAP 512차원, 1초 hop, fp16 | 약 10GB / 약 40GB |

원본 영상은 `data/chapters_audio/tmp_media`에 잠깐 머물다 삭제된다(최대 100GB).

**추출 전에 확정된 설정(바꾸면 전부 다시 추출)**: CLIP 2fps, CLAP 1초 window/hop, 512차원, fp16, LMDB.

```bash
# 3-5. 최종 리스트 확정: 두 LMDB에 모두 있는 id만 남기고, Val 300 / Test 1,000개를 고정하고, 교집합 0을 검증
bash scripts/audio/data_pipeline.sh finalize
#   -> subsets/{train_sub,val_sub,test_sub}.json  (이후 모든 행이 이 파일을 읽는다)
```

`finalize`는 `--train_anno/--test_anno/--val_anno`로 4단계의 annotation JSON도 필터링하므로,
4단계를 먼저 돌린 뒤 `finalize`를 실행해도 된다(순서는 4 → 3-5 권장).

---

## 4. Annotation 변환 + query 텍스트 feature

ReVisionLLM은 activitynet 형식 annotation과 CLIP 텍스트 feature LMDB를 읽는다.

```bash
# 4-1. 학습용 (Train-sub id만, CLIP LMDB에 있는 영상만)
python revisionllm/data/vidchap7m/chapters_to_activitynet.py \
    --chapters_data_path data/chapters/chapters_vmr_train.jsonl \
    --chapters_out_path data/chapters/chapters_train.json \
    --feat_lmdb data/chapters_audio/clip_l14_2fps_lmdb

# 4-2. 평가용 test / val
python revisionllm/data/vidchap7m/chapters_test_to_activitynet.py \
    --chapters_data_path data/chapters/chapters_vmr_test.jsonl \
    --chapters_out_path data/chapters/chapters_test.json
python revisionllm/data/vidchap7m/chapters_test_to_activitynet.py \
    --chapters_data_path data/chapters/chapters_vmr_val.jsonl \
    --chapters_out_path data/chapters/chapters_val.json

# 4-3. query 텍스트 feature (CLIP ViT-L/14 text). 키 규칙이 학습/평가에서 다르다.
cd revisionllm/data/feature_extraction
python chapters_clip_text_extractor.py --chapters_dir ../../../data/chapters --splits train \
    --key_style vid_idx --clip_path ../../../checkpoints/clip/ViT-L-14.pt \
    --feature_output_path ../../../data/chapters_audio/clip_L14_text_features_train
python chapters_clip_text_extractor.py --chapters_dir ../../../data/chapters --splits test,val \
    --key_style qid --clip_path ../../../checkpoints/clip/ViT-L-14.pt \
    --feature_output_path ../../../data/chapters_audio/clip_L14_text_features_test
cd ../../..

# 4-4. (3-5를 아직 안 했다면 여기서) finalize -> subsets/{train,val,test}_sub_anno.json 생성
bash scripts/audio/data_pipeline.sh finalize
```

---

## 5. Test-sub query 유형 라벨 (H2 breakdown용, 선택이지만 권장)

```bash
export ANTHROPIC_API_KEY=...          # 없으면 NO_LLM=1 로 키워드 라벨만 생성
bash scripts/audio/data_pipeline.sh labels
#   -> subsets/test_query_types.json  (키워드와 Claude 판단이 일치하는 query만 breakdown에 사용)
```

---

## 6. 검증: 단위 테스트와 R0 재현 (학습 전에 반드시)

```bash
# 6-1. 오디오 branch 단위 테스트 (CPU, 수 초)
python -m pytest tests/test_audio_fusion.py -q

# 6-2. R0: 공개 체크포인트를 Test-sub에서 평가. 논문 수치와 비교해 파이프라인이 맞는지 확인.
AUDIO=0 bash scripts/audio/eval_row.sh checkpoints/public/chapters_stage2_long_100 R0
#   -> checkpoints/audio/eval/R0/predictions_merged.txt + result_retrieval.txt (mIoU, R1@k)

# 6-3. (선택) 공개 체크포인트에 오디오 branch를 붙여도 결과가 R0와 같은지 확인 (zero-init identity)
bash scripts/audio/eval_row.sh checkpoints/public/chapters_stage2_long_100 R0_audio_identity
```

R0의 R1@0.5가 논문 대비 3포인트 이상 낮으면 feature fps, window, prompt를 먼저 의심한다.

---

## 7. 학습

> 아래 소요 시간은 RTX 4090 기준 추정치다. 3090(gpu4)은 더 느리다. 첫 Stage A에서 step당 시간을 재서 갱신할 것.

행(row) 정의: **R1** = visual-only 계속 학습(baseline), **R2** = audio-visual(제안).
시드는 `SEED=0|1|2`로 준다. 출력은 `checkpoints/audio/stage{A,B,C33,C100}_{R1,R2}_seed{N}/`.

```bash
# 7-1. Stage A (R2만): 오디오 branch warm-up, 약 4시간, 1 job
SEED=0 bash scripts/audio/stage_a.sh
#   gate 초기값은 0.1이다(GATE_INIT 환경변수로 변경). 0.0이면 gate와 fc_out이 모두 0이라 모든 오디오
#   파라미터의 gradient가 정확히 0이 되어 학습이 시작되지 않는다. step 0 출력은 어느 쪽이든 identity.
#   로그에서 audio_fusion gate 값이 200 step 안에 0.05 이상으로 움직이는지 확인.
#   안 움직이면: --audio_lr_multiplier 30 으로 재시작.

# 7-2. 분할 재개 검증 (한 번만): Stage A를 도중에 죽인 뒤 같은 명령으로 재개해 loss 곡선이 이어지는지 확인
MAX_TRAIN_HOURS=0.5 SEED=0 bash scripts/audio/stage_a.sh   # 30분 뒤 스스로 멈춤
SEED=0 bash scripts/audio/stage_a.sh                          # 마지막 checkpoint-*에서 자동 재개

# 7-3. Stage B: 약 1~1.5일 => 같은 명령을 2번 제출
SEED=0 bash scripts/audio/stage_b.sh R2      # Stage A 출력에서 시작, modality dropout 15/15
SEED=0 bash scripts/audio/stage_b.sh R2      # 재개 (완료돼 있으면 "already holds final weights"로 즉시 종료)
SEED=0 bash scripts/audio/stage_b.sh R1      # 공개 체크포인트에서 시작, 오디오 없음, 같은 step/시드
SEED=0 bash scripts/audio/stage_b.sh R1
# 시드 1, 2도 동일하게 (R1·R2 x 3 시드 = 6 run, 약 7일)

# 7-4. Val-sub로 시드 선택 (Stage C는 시드 1개만)
#   각 Stage B 출력에 대해 TEST_ANNO/TEST_IDS를 val_sub로 바꿔 eval_row.sh 실행
TEST_ANNO=data/chapters_audio/subsets/val_sub_anno.json TEST_IDS=data/chapters_audio/subsets/val_sub.json \
  EVAL_ROOT=checkpoints/audio/eval_val bash scripts/audio/eval_row.sh checkpoints/audio/stageB_R2_seed0 B_R2_seed0

# 7-5. Stage C-33 (약 12시간, 1 job) -> C-100 (약 1일, 2 job)
SEED=0 bash scripts/audio/stage_c.sh R2 33
SEED=0 bash scripts/audio/stage_c.sh R2 100
SEED=0 bash scripts/audio/stage_c.sh R2 100
SEED=0 bash scripts/audio/stage_c.sh R1 33
SEED=0 bash scripts/audio/stage_c.sh R1 100
SEED=0 bash scripts/audio/stage_c.sh R1 100
```

Stage C는 alternate hierarchy 모드라 `dataloader_num_workers 0`이 강제된다(스크립트에 고정). 느리면
`SPARSE_LENGTH`(기본 2500초, 긴 영상만 사용)나 `MAX_STEPS`를 조정한다.

---

## 8. 평가 (행당 약 4시간)

```bash
E=checkpoints/audio
AUDIO=0 bash scripts/audio/eval_row.sh $E/stageC100_R1_seed0 R1_seed0
        bash scripts/audio/eval_row.sh $E/stageC100_R2_seed0 R2_seed0
DROP_VISUAL=1 bash scripts/audio/eval_row.sh $E/stageC100_R2_seed0 R3     # audio-only 추론 (H3)
DROP_AUDIO=1  bash scripts/audio/eval_row.sh $E/stageC100_R2_seed0 R4     # 시각 경로 훼손 여부
# 보조 행 (시드 0만)
#   R5: MD_VISUAL=0 MD_AUDIO=0 로 stage_b.sh R2 -> stage_c.sh -> eval  (modality dropout 없음)
#   R6: Stage A 생략: INIT=$PUBLIC_STAGE2 SEED=0 bash scripts/audio/stage_b.sh R2 -> stage_c.sh -> eval
#   R7: Stage B 출력만 평가: bash scripts/audio/eval_row.sh $E/stageB_R2_seed0 R7
```

각 행의 출력: `checkpoints/audio/eval/<행>/predictions_merged.txt`(query별 IoU), `dense/result_retrieval.txt`.

---

## 9. 리포트

```bash
bash scripts/audio/report.sh
#   -> checkpoints/audio/eval/ablation_report.json
#   콘솔: 행별 mIoU/R1@0.3/0.5/0.7 (시드 평균±표준편차), R2−R1 paired bootstrap 95% CI, query 유형별 breakdown
```

CI가 0을 포함하지 않으면 H1 성립. 평균 이득이 작아도 `audio_cue` 그룹에서 이득이 있으면 H2 성립.

---

## 디렉터리 요약

```
checkpoints/
  vicuna-7b-v1.5/  clip/ViT-L-14.pt  vtimellm-vicuna-v1-5-7b-stage1/  public/chapters_stage{1_sparse,2_long_100}/
  audio/stage{A,B,C33,C100}_{R1,R2}_seed{N}/     # 각 run: checkpoint-*/ (재개용), adapter_*, non_lora_trainables.bin (완료 표시)
  audio/eval/<행>/                               # dense/, e2e2/, predictions_merged.txt
data/chapters/            chapters_vmr_*.jsonl, chapters_{train,test,val}.json
data/chapters_audio/      clip_l14_2fps_lmdb/, clap_1s_lmdb/, clip_L14_text_features_{train,test}/, subsets/, tmp_media/
```

## 자주 겪을 문제

- **`already holds final weights`**: 그 run은 끝난 것. 다시 돌리려면 출력 폴더를 지운다.
- **OOM**: `env.sh`의 배치는 2/1이다. 그래도 넘치면 학습 스크립트 인자에 `--bits 4`를 추가(QLoRA). 이 경우 ablation 모든 행을 같은 설정으로 다시 맞춘다.
- **Stage A에서 gate가 0 근처**: 위 7-1 참고. `GATE_INIT=0.0`으로 돌린 것이 아닌지 먼저 확인한다(gradient가 0이라 영원히 안 움직임). `--modality_dropout_visual`을 Stage A에서도 0.3으로 켜보는 것이 다음 수단.
- **다운로드 실패가 많음**: `extract_*.jsonl`의 `download_failed` 비율을 보고, Train-sub 목표 시간(`TRAIN_HOURS`, 기본 2500)을 낮춰 `subsets`부터 다시 한다. Test-sub는 후보를 1,500개 뽑아두므로 웬만하면 1,000개가 채워진다.
- **feature_fps 불일치**: 모든 스크립트가 2fps를 가정한다. 다른 fps로 추출했다면 `env.sh`의 `FEATURE_FPS`를 바꾸되, 공개 체크포인트는 2fps 기준이므로 R0가 깨진다.
