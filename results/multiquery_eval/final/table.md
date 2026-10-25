| 구성 | mIoU | R1@0.3 | R1@0.5 | R1@0.7 | 평균 지연(s/query) | feature 추출 | grounding 호출 | cache hit |
|---|---|---|---|---|---|---|---|---|
| Baseline | 52.72 | 72.99 | 61.92 | 38.69 | 2.54 | 822 | 822 | 0 |
| + Feature Cache | 53.08 | 73.24 | 62.65 | 39.05 | 1.29 | 300 | 822 | 0 |
| + Semantic Cache | 53.12 | 73.24 | 62.17 | 38.69 | 1.22 | 300 | 786 | 36 |

822 queries / 300 videos. semantic hits: 36 of 822 queries; on those, mean IoU 0.496 (reused) vs 0.492 (Baseline grounding)
