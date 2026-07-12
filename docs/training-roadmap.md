# Training Roadmap

MedFact-Bench is held-out evaluation data. It must not be used for supervised or
reinforcement learning. Future training support will use MedFact-Synth through a
separate taskset in this repository.

## Planned v0.2.0 Components

- A pinned MedFact-Synth loader with deterministic train and validation splits.
- A chat-format exporter for supervised fine-tuning.
- A five-point RLVR task and paper-faithful rule-based reward.
- Prime RL configuration profiles for baseline, SFT, RLVR, and post-training evaluation.
- Run manifests containing dataset, model, dependency, hardware, and checkpoint revisions.

## Paper Training Contract

The Med-V1 publication describes a two-stage procedure:

1. SFT trains the model to produce the synthetic rationale and five-point score.
2. RLVR initializes from the SFT checkpoint and applies GRPO with five rollouts per prompt.
3. Invalid format or an invalid score receives a reward of `-1`.
4. A valid score receives `0.5 * (2 - abs(predicted_score - ground_truth_score))`.
5. RL uses a 3,072-token prompt limit and a 1,024-token response limit.

## Reproduction Decision Gate

The publication and the released Med-V1 training scripts do not currently agree on
every hyperparameter. For example, the paper reports an RL learning rate of `1e-3`
and a global batch size of `1,440`, while the released GRPO shell script specifies
`1e-6` and a training batch size of `480` with five rollouts. These sources must be
reconciled and recorded as separate named profiles before training code is released.
No profile may silently combine values from different sources.

