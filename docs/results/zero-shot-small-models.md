# Zero-Shot Small Model Results

This page reports local zero-shot MedFact-Bench runs for the two small models from
Figure 3 of the Med-V1 paper. The runs use the public `medfact-bench` environment
without prompt, parser, dataset, or metric modifications.

Prime evaluation pages are included for rollout inspection. The metric source of
truth is the JSON report artifact beside this document.

## Setup

| Field | Value |
| --- | --- |
| Dataset | `ncbi/MedFact-Bench` |
| Dataset revision | `249028caf7ad5a3e63331269a606f4b2696693ed` |
| Examples | `14,274` |
| Subset | `all` |
| Rollouts per example | `1` |
| Temperature | `0` |
| Max tokens | `1024` |
| Backend | `vLLM` |
| Hardware class | single NVIDIA RTX 4070 12GB GPU |
| Comparison metric | unweighted five-dataset macro accuracy |

Endpoint URLs, hostnames, local filesystem paths, credentials, and raw traces are
not included in this public artifact.

## Summary

| Model | Micro Acc | Macro Acc | Paper Macro | Delta | Macro F1 | Parse Rate | Invalid | Prime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen/Qwen2.5-3B-Instruct | 0.5385 | 0.5200 | 0.515 | +0.0050 | 0.4529 | 0.9018 | 1402 | [Prime](https://app.primeintellect.ai/dashboard/evaluations/uj1uugk1z7kl2f194s2x5dgc) |
| meta-llama/Llama-3.2-3B-Instruct | 0.4953 | 0.5129 | 0.511 | +0.0019 | 0.4120 | 0.9021 | 1398 | [Prime](https://app.primeintellect.ai/dashboard/evaluations/c52x83cg9z2h91p72orwnx8e) |

## Per-Dataset Accuracy

### Qwen/Qwen2.5-3B-Instruct

| Dataset | Correct | Total | Accuracy | Paper Accuracy | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| scifact | 213 | 340 | 0.6265 | 0.638 | -0.0115 |
| healthver | 420 | 903 | 0.4651 | 0.447 | +0.0181 |
| medaesqa | 5292 | 9106 | 0.5812 | 0.577 | +0.0042 |
| pubmedqa-fact | 242 | 500 | 0.4840 | 0.470 | +0.0140 |
| bioasq-fact | 1519 | 3425 | 0.4435 | 0.444 | -0.0005 |

### meta-llama/Llama-3.2-3B-Instruct

| Dataset | Correct | Total | Accuracy | Paper Accuracy | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| scifact | 198 | 340 | 0.5824 | 0.600 | -0.0176 |
| healthver | 373 | 903 | 0.4131 | 0.435 | -0.0219 |
| medaesqa | 4420 | 9106 | 0.4854 | 0.437 | +0.0484 |
| pubmedqa-fact | 279 | 500 | 0.5580 | 0.550 | +0.0080 |
| bioasq-fact | 1800 | 3425 | 0.5255 | 0.531 | -0.0055 |

## Artifacts

- [`qwen2.5-3b-instruct-medfact-report.json`](qwen2.5-3b-instruct-medfact-report.json)
- [`llama-3.2-3b-instruct-medfact-report.json`](llama-3.2-3b-instruct-medfact-report.json)
- [`run-manifest.json`](run-manifest.json)

## Notes

- `Micro Acc` is equivalent to Prime `avg_reward` for these runs.
- `Macro Acc` is the unweighted mean of the five component dataset accuracies and
  is the appropriate comparison to the Med-V1 paper.
- Invalid predictions count as incorrect and remain in all denominators.
