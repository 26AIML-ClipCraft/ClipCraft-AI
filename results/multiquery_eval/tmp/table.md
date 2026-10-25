| 구성 | mIoU | R1@0.3 | R1@0.5 | R1@0.7 | 평균 지연(s/query) | feature 추출 | grounding 호출 | cache hit |
|---|---|---|---|---|---|---|---|---|
| Baseline | 59.66 | 75.00 | 62.50 | 62.50 | 23.37 | 8 | 8 | 0 |
| + Feature Cache | 69.86 | 87.50 | 75.00 | 75.00 | 12.83 | 3 | 8 | 0 |
| + Semantic Cache | 70.58 | 87.50 | 87.50 | 75.00 | 11.96 | 3 | 7 | 1 |

8 queries / 3 videos. semantic hits: 1 of 8 queries; on those, mean IoU 0.907 (reused) vs 0.907 (Baseline grounding)
