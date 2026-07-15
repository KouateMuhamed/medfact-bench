# MedFact-Bench

MedFact-Bench is a zero-shot environment for biomedical claim verification and
evidence attribution. It implements the protocol introduced in
[Med-V1: Small Language Models for Zero-shot and Scalable Biomedical Evidence Attribution](https://arxiv.org/abs/2603.05308)
using the native Verifiers 0.2 Taskset/Harness API.

- **Environment ID:** `medfact-bench`
- **Task:** single-turn, three-class medical claim verification
- **Tags:** medical, fact-verification, evidence-attribution, evaluation
- **Maintainer:** Kouate Muhamed ([KouateMuhamed](https://github.com/KouateMuhamed))

## Dataset

- **Dataset:** [`ncbi/MedFact-Bench`](https://huggingface.co/datasets/ncbi/MedFact-Bench)
- **Pinned revision:** `249028caf7ad5a3e63331269a606f4b2696693ed`

| Component | Examples |
| --- | ---: |
| SciFact | 340 |
| HealthVer | 903 |
| MedAESQA | 9,106 |
| PubMedQA-Fact | 500 |
| BioASQ-Fact | 3,425 |
| **Total** | **14,274** |

The loader preserves row order and duplicate rows. It validates all required fields,
component names, labels, and the shared system prompt. A local Parquet file can be
used instead of Hugging Face by setting `taskset.dataset_path`.

MedFact-Bench is exposed only as an evaluation split. Training requests fail rather
than silently using held-out benchmark examples.

## Protocol

Each example contains a source article and a medical claim. The model returns a
five-point evidence-attribution score, which maps to the benchmark labels as follows:

| Score | Prediction |
| ---: | --- |
| `1`, `2` | `SUPPORT` |
| `0` | `NEI` |
| `-1`, `-2` | `CONTRADICT` |

The parser reads the first `<score>...</score>` occurrence and accepts only an integer
from `-2` through `2`. Missing, malformed, and out-of-range scores are classified as
`INVALID`, receive zero accuracy, and remain in all reporting denominators.

### System Prompt

The dataset-provided system prompt is passed to the model unchanged:

```text
You are a fact-checking expert trained in evidence-based medicine. Your task is to evaluate how strongly an *article* agrees or disagrees with a *claim*. The *article* is retrieved from a search engine using the *claim* as the query.

Use the following five-point scale:
   - **-2 Strong Contradiction**  – The article clearly and directly refutes the claim.
   - **-1 Partial Contradiction** – The article provides mixed or indirect evidence against the claim.
   - ** 0 Neutral / Unrelated**   – The article does not address the claim, offers insufficient information, or is irrelevant to the claim.
   - ** 1 Partial Agreement**	 – The article offers some indirect or tentative support for the claim.
   - ** 2 Strong Agreement**	 – The article explicitly and strongly supports the claim.

Note that the *article* might not describe the exact same subjects, interventions, or measurements as the *claim*. In this case, please note the difference and assign a score of 0.

Output in two parts only and do not output anything else:
<think>[your detailed, step‐by‐step explanation for scoring]</think>
<score>[the integer score only, i.e., -2, -1, 0, 1, or 2]</score>
```

The square brackets identify placeholder content. They are not required in model output.

### User Prompt

```text
Article:
{article}

Claim:
{claim}
```

## Installation

Install the local checkout:

```bash
uv sync
```

Inside a Prime Lab workspace, clone the repository into the underscore-normalized
environment directory before using the local environment installer:

```bash
git clone https://github.com/KouateMuhamed/medfact-bench.git environments/medfact_bench
prime env install medfact-bench --path environments --prerelease
```

After the first public release, install the immutable Hub version:

```bash
prime env install KouateMuhamed/medfact-bench@0.1.0 --prerelease
```

## Evaluation

Run the reproducible zero-shot configuration:

```bash
uv run eval @ configs/eval/zero-shot.toml
```

Run a small evaluation through the Prime CLI:

```bash
prime eval run medfact-bench \
  -m "openai/gpt-5-mini" \
  -n 5 \
  -r 1
```

Verifiers 0.2 configuration fields are namespaced under the taskset:

```toml
[taskset]
id = "medfact-bench"
subset = "scifact"
split = "eval"
dataset_path = "/absolute/path/to/MedFact-Bench.parquet"

[harness]
id = "medfact-bench"
```

Inference model, endpoint, concurrency, and hardware are caller-owned. Sampling for
paper comparisons should use one rollout, temperature `0`, and at most 1,024 output tokens.

## Metrics

| Metric | Role | Meaning |
| --- | --- | --- |
| `accuracy` | reward, weight 1.0 | Exact three-class accuracy |
| `parseable_score` | metric | Whether a valid five-point score was parsed |
| `strict_format` | metric | Exact compliance with one think block followed by one score block |

The average reward is example-weighted micro accuracy. The paper's primary metric is
the unweighted mean of the five component accuracies.

## Reporting

Verifiers 0.2 writes `traces.jsonl`. Generate the benchmark report with either command:

```bash
medfact-report outputs/path/to/traces.jsonl
python -m medfact_bench.reporting outputs/path/to/traces.jsonl
```

The reporter also accepts legacy `results.jsonl`. For legacy runs it updates a sibling
`metadata.json` non-destructively when present. Use `--no-update-metadata` to disable
that behavior.

The versioned report includes parse and format rates, micro and macro accuracy,
per-component accuracy, per-class precision/recall/F1, macro F1, a confusion matrix
including `INVALID`, and coverage diagnostics. `paper_macro_accuracy` is populated only
when all 14,274 examples have exact component coverage.

## Published Results

Figure 3 of the Med-V1 paper reports the following zero-shot accuracies.

### Small Models

| Model | SciFact | HealthVer | MedAESQA | PubMedQA-Fact | BioASQ-Fact | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-3B-Instruct | .600 | .435 | .437 | .550 | .531 | .511 |
| Qwen2.5-3B-Instruct | .638 | .447 | .577 | .470 | .444 | .515 |

### Frontier Models

| Model | SciFact | HealthVer | MedAESQA | PubMedQA-Fact | BioASQ-Fact | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.3-70B-Instruct | .812 | .597 | .687 | .774 | .771 | .728 |
| o3-mini | .832 | .591 | .733 | .778 | .747 | .736 |
| GPT-4o-mini | .847 | .580 | .704 | .734 | .720 | .717 |
| GPT-4o | .832 | .576 | .717 | .776 | .777 | .736 |
| GPT-5 | .818 | .615 | .703 | .788 | .753 | .735 |

### Fine-Tuned Models

| Model | SciFact | HealthVer | MedAESQA | PubMedQA-Fact | BioASQ-Fact | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Med-V1-L3B | .844 | .575 | .748 | .746 | .725 | .728 |
| Med-V1-Q3B | .856 | .588 | .733 | .764 | .717 | .732 |

Macro is the unweighted mean of the five component accuracies. Evaluation manifests
should preserve dataset, model, dependency, prompt, and sampling revisions.

Local small-model replication results are available in
[`docs/results/zero-shot-small-models.md`](docs/results/zero-shot-small-models.md).

## Next Steps

This first release establishes the zero-shot MedFact-Bench evaluation workflow.
Future work will extend the project toward the Med-V1 training pipeline.

- [x] Implement the zero-shot MedFact-Bench environment.
- [x] Add benchmark-specific reporting and result manifests.
- [x] Replicate the paper's small-model zero-shot baselines locally.
- [ ] Add MedFact-Synth data preparation for supervised fine-tuning.
- [ ] Evaluate SFT checkpoints with the same MedFact-Bench protocol.
- [ ] Implement the paper's RLVR reward and GRPO training setup.
- [ ] Compare zero-shot, SFT, and RLVR checkpoints with reproducible manifests.

## Citation

If this environment supports research, cite both this software and the Med-V1 paper.
Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Author

Kouate Muhamed  
Email: [muhamed.kouate@icloud.com](mailto:muhamed.kouate@icloud.com)

## License

This environment is released under the MIT License. Dataset and upstream attribution
are documented in [`NOTICE`](NOTICE).
