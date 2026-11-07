# 시나리오 추천 + 하이라이트 추출 (공유 audio-visual 표현)

`multiquery`의 `FeatureCache`가 만든 CLIP ViT-L/14 + CLAP feature를 1초 격자의 `AVRepresentation`으로 묶어
Scenario Grounding / Scenario Recommendation / Highlight Extraction이 함께 읽는다. 영상 인코딩은 파일당 한 번이다.

| 파일 | 역할 |
| --- | --- |
| `shared.py` | 캐시 feature → 1초 격자 `AVRepresentation` |
| `univtg_head.py` | UniVTG 구조(saliency / foreground / boundary offset)를 CLIP-L + CLAP 위에 재구현. 손실·계수는 레퍼런스(`showlab/UniVTG`)를 따름 |
| `dataset.py`, `train_head.py` | 캐시 LMDB만 읽어 학습(추출 없음). 챕터 구간·제목에서 UniVTG식 unified label 생성 |
| `audio_events.py` | CLAP zero-shot 이벤트(speech/music/laughter/applause/ambient/silence) |
| `scoring.py` | saliency + 오디오 이벤트 + 시각 변화 → 초당 중요도 |
| `summarize.py` | 변화점 분할 + MMR로 대표 구간 선택(시간, 오디오 태그, 키프레임) |
| `scenario_llm.py` | gpt-4o-mini: 구간 캡션(이미지) → 편집 순서/시나리오 JSON. 존재하지 않는 구간은 버림 |
| `highlight.py` | 중요도 피크로 클립 선택 후 ffmpeg로 이어붙임 |
| `query_gen.py`, `qwen_describe.py` | 제목이 없을 때 로컬 Qwen3-VL-8B(기본, `--vlm_model`로 Qwen2.5-VL 선택 가능)가 프레임 8장으로 한 문장 설명을 만들어 헤드의 텍스트 쿼리로 사용(영상당 1회, DB 캐시) |
| `db.py`, `service.py`, `cli.py` | `edit_result` 테이블 캐시, 오케스트레이션, CLI |

## 사용법
```bash
export OPENAI_API_KEY=...  CLIPCRAFT_DB_DSN=postgresql://localhost:5432/clipcraft
python revisionllm/editing/cli.py recommend --video X.mp4 --goal "1분 요약"
python revisionllm/editing/cli.py highlight --video X.mp4 --out X_hl.mp4 --title "제주 여행 브이로그"   # 제목이 없으면 Qwen이 설명 생성
python revisionllm/editing/cli.py all --video X.mp4 --query "a person laughs" --out X_hl.mp4   # 7B 모델 로드
```
헤드 학습(GPU, 아직 실행하지 않음):
```bash
python revisionllm/editing/train_head.py --train_anno data/chapters/chapters_vmr_train.jsonl \
  --val_anno data/chapters/chapters_vmr_val.jsonl --clip_lmdb data/chapters_audio/clip_l14_2fps_lmdb \
  --clap_lmdb data/chapters_audio/clap_1s_lmdb --out checkpoints/editing_head/head.pt
```

## UniVTG 레퍼런스와의 대응 (레포를 읽고 맞춘 부분)
- **하이라이트 점수** = saliency 점수 + foreground 로짓 (`main/inference_hl.py`, `eval_mode=add`). 여기에 오디오 이벤트·시각 변화를 섞는 것은 이 프로젝트의 확장.
- **요약 점수** = foreground 로짓만 (`main/inference_qfvs.py`, `pred_logits`). 레퍼런스는 상위 클립을 바로 뽑지만 여기서는 변화점으로 나눈 구간 단위로 고르고(MMR) 시간·장면 내용을 구조화한다.
- **손실** (`scripts/qvhl_pretrain.sh`): foreground = 배경 가중 0.1의 교차엔트로피(계수 10), saliency = 구간 안/밖 클립 쌍의 margin 1 랭킹(계수 0.05), 경계 = L1(10) + gIoU(1). 라벨은 `get_saliency_labels_sub_as_query`처럼 정답 구간에서 만든다.
- **차이:** 레퍼런스는 사람이 만든 saliency 라벨과 영상 제목 쿼리로 학습한다. 여기에는 둘 다 없어서 영상 제목 대신 챕터 제목들을 이은 문장을 쿼리로 쓰고, 그 샘플은 챕터 제목과 가장 잘 맞는 상위 30% 초를 foreground로 쓴다(약한 대리 라벨). 경계 offset은 학습만 하고 두 기능에서는 쓰지 않는다(레퍼런스도 MR에서만 사용).

## 쿼리 (헤드가 보는 텍스트)
- 학습: 챕터 제목(70%, 구간 정답이 있는 쿼리) 또는 챕터 제목들을 이은 문장(30%, 영상 제목 대용). 쿼리 없는 모드는 두지 않는다.
- 추론: `--title`이 있으면 그것, 없으면 Qwen3-VL-8B(`~/.cache/huggingface`)가 쓴 한 문장. Qwen3-VL은 transformers>=4.57이 필요해
  이 환경(4.41.2) 대신 다른 환경의 python으로 서브프로세스 실행한다: `export CLIPCRAFT_VLM_PYTHON=~/miniconda3/envs/gazeqwen/bin/python`.
- 헤드 체크포인트가 없으면 쿼리는 쓰지 않고 VLM도 부르지 않는다.

## 한계
- 체크포인트가 없으면 saliency는 "영상 평균과의 유사도"(training-free)로 대체되고 결과에 `saliency_source`로 표시된다.
- 학습 데이터에는 사람이 만든 하이라이트 라벨이 없다. saliency 타깃은 챕터 제목-프레임 CLIP 유사도라 약한 대리 신호이며,
  그래서 최종 중요도는 오디오 이벤트·시각 변화와 섞는다(`EditConfig.w_*`). 품질은 학습 후 사람이 확인해야 한다.
- 결과 캐시 키에 설정·feature_version·헤드 버전이 들어가므로 바뀌면 재계산된다. LLM 캐시는 모델/언어까지 포함.
- 테스트: `pytest tests/test_editing.py` (DB 테스트는 `CLIPCRAFT_TEST_DSN` 필요). 학습·실영상·실제 GPT 호출은 검증하지 못했다.
