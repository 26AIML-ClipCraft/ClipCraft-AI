# 다중 쿼리 Grounding + Feature / Query 캐시

학습된 **R2** 체크포인트(`checkpoints/audio_charades_long/stageB_R2_seed0`, 오디오-비주얼, dense 경로 `iteration_step=1`)를
모델 코드 수정 없이 사용한다. 기존 코드를 그대로 재사용한 부분: `ClipFrameExtractor`, `ClapAudioExtractor`,
`ClipFeatureExtractor.encode_text`, `AudioStore.align`, `inference`, 그리고 `eval_nlq_negative.py`의
`iou` / `questions` / `parse_args` (윈도우 구성과 후보 점수화도 `eval_nlq_negative.eval`과 동일).

| 파일 | 역할 |
| --- | --- |
| `feature_cache.py` | SHA-256 → `video` 행 → `vision.npy` / `audio.npy` (쿼리와 무관한 feature, 파일당 한 번만 추출) |
| `db.py` | PostgreSQL + pgvector 스키마: `video (1) → (N) query_cache` |
| `grounder.py` | R2를 한 번만 로드, 영상의 윈도우를 한 번만 만들어 모든 쿼리가 공유, 쿼리별 grounding |
| `service.py` | 쿼리마다 CLIP 텍스트 임베딩 → 같은 `video_id` 안에서 코사인 검색 → 히트: 재사용 / 미스: grounding 후 저장 |
| `cli.py` | `--video X.mp4 --query ... --query ...` |

## 사용법

```bash
export CLIPCRAFT_DB_DSN=postgresql://localhost:5432/clipcraft      # DB는 미리 생성. pgvector extension은 첫 실행 때 자동 생성
python revisionllm/multiquery/cli.py --video data/charades/Charades_v1_480/AL1WC.mp4 \
    --query "the person closes the door." --query "a person is sneezing"
```

## 설계 메모

- **버전 관리.** `video.feature_version`(extractor 이름, fps, hop, `CacheConfig.feature_schema`)이 다르면 feature를
  다시 추출하고 그 영상의 쿼리 캐시를 삭제한다. `query_cache.model_version`(체크포인트 디렉터리 + adapter 크기/mtime)은
  검색 조건에 포함되므로 다른 체크포인트의 결과는 재사용되지 않는다.
- **Threshold** (`CacheConfig.similarity_threshold`, `--threshold`, 기본 0.95, 엄밀히 초과일 때 히트). 임베딩은 모델이 쓰는
  CLIP 텍스트 EOT feature다. Charades-STA 테스트의 같은 영상 쿼리 쌍 5,527개에서 0.95로 히트한 경우 98.0%가 같은 장면
  (GT IoU ≥ 0.5)을 가리켰고 전체 쌍의 3.6%가 히트했다. 0.90은 90.7% / 11.4%. 임베딩이나 데이터 도메인이 바뀌면
  threshold를 다시 검증할 것(검증 스크립트는 `audio-fusion` 브랜치).
- **쿼리 배치.** forward 한 번의 행은 (쿼리, 윈도우) 쌍이며 모든 쿼리가 같은 윈도우 텐서를 공유한다. 행마다 프롬프트와
  쿼리 feature가 다르고, 길이가 다른 프롬프트는 **left padding**한다(`inference()`가 프롬프트 리스트를 받음, 기존 단일
  프롬프트 호출은 그대로). 패딩을 지원하려고 모델 코드 두 곳을 고쳤다.
  - `vtimellm_llama.py` `prepare_inputs_for_generation`: 첫 step 이후 attention mask를 전부 1로 다시 만들던 것을, 첫 step에서
    `<video>`가 확장된 프롬프트의 mask(`forward`가 `_gen_attention_mask`로 보관)를 이어 쓰도록 변경. 패딩이 없으면 mask가 전부
    1이라 기존과 동일하다. position_ids는 `vtimellm_arch.py`가 mask 합으로 이미 계산한다.
  - `Grounder`가 `config.tokenizer_padding_side = "left"`를 설정(`vtimellm_arch.py`에 있던 옵션 사용).
  - 검증(CPU fp32, AL1WC): 길이 19/18/20/19 쿼리 4개를 한 배치로 돌린 결과가 순차 실행과 같은 범위였고, 패딩된 행의 처음
    6 step logit이 단독 실행과 정확히 일치(차이 0.0). mask 수정을 끈 대조군은 6~12 차이가 났다.
  - 한계: `_gen_attention_mask`는 모델 인스턴스 상태라 같은 모델에서 `generate`를 동시에 여러 스레드로 호출하면 안 된다.
    bf16 GPU에서는 배치 크기/패딩에 따라 수치가 미세하게 달라질 수 있다. 속도 이득은 GPU가 없어 측정하지 못했다.
  한 패스의 최대 행 수는 `--query_batch`(기본 8)로 GPU 메모리를 제한한다. 행은 프롬프트 길이순으로 정렬해 패딩을 줄인다.
- **한 요청 안의 중복 쿼리.** 이미 캐시에 없는 쿼리끼리도 유사도가 threshold를 넘으면 앞 쿼리 결과를 따라가며
  grounding은 한 번만 한다.
- **CLAP 비결정성.** transformers의 fused CLAP 전처리가 윈도우마다 mel 구간을 무작위로 뽑아 실행마다 일부 윈도우가 달라진다.
  재추출 재현성을 위해 추출 직전에 `np.random.seed(0)`을 설정한다.
- dense 단일 단계 경로(R2 평가에 쓰인 것)만 구현했다. `eval_nlq_retrieval_e2e2.py`의 윈도우 검색 단계는 윈도우가 약 100개
  이상인 긴 영상에서만 동작하므로(`eval_row.sh` 참고) 사용하지 않았다.

## 테스트

가짜 extractor/encoder/grounder + 실제 PostgreSQL:

```bash
CLIPCRAFT_TEST_DSN=postgresql://localhost:5433/clipcraft pytest tests/test_multiquery_cache.py
```
