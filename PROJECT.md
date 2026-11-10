# CPS 375 / 575 Mini-Project: Multi-Task LLM Fine-Tuning

For setup, evaluation commands, and getting started, see [`README.md`](README.md).

## Overview

In this project, you will post-train the pretrained base model **`Qwen/Qwen3.5-9B-Base`** into **one checkpoint** that does well on three tasks at once: instruction following, math reasoning, and Python code generation. 

You will train with **Tinker**, a training API: you write the data pipeline and training loop in Python, and Tinker runs them on its GPUs. Tinker uses **LoRA**, which trains a small low-rank update instead of all the model's weights, allowing fast and cheap finetuning.

Each task has a public benchmark that you can evaluate your model:

- **IFEval** for instruction following
- **MATH-500** for math reasoning
- **MBPP** for code generation

Use these benchmarks to measure progress, and improve all three **jointly**, since gains on one task can come at the cost of another. Aim to improve the tasks themselves rather than these specific benchmarks: we will also evaluate your checkpoint on **hidden held-out benchmarks** for the same tasks.

| | **CPS 375** | **CPS 575** |
|---|---|---|
| Core work | SFT, including data selection and mixture tuning | The same SFT work, **plus the [CPS 575 component](#the-cps-575-component)**: an extension beyond SFT, or an approved open-ended project |
| Ranked against | Other **CPS 375** teams | Other **CPS 575** teams |

**Key dates:** open-ended proposals (**CPS 575**, optional) are due **November 18, 11:59 PM**; final submissions are due **December 11, 11:59 PM**.

## Model and budget

Your graded checkpoint must be fine-tuned from `Qwen/Qwen3.5-9B-Base`. Other models on Tinker may be used as teachers (for example, for distillation), but not as the base of the graded checkpoint.

Each team member has a **$250 Tinker budget**. For reference, one full evaluation on all three public benchmarks costs about $0.60, and an SFT run on about 25,000 examples costs about $12. Use `--limit` for quick checks during development, and record your approximate costs.

## Benchmarks and thresholds

| Skill | Public benchmark | Samples | Metric | Threshold |
|---|---|---:|---|---:|
| Instruction following | IFEval | 541 | Average of prompt-level and instruction-level accuracy, strict and loose | 58.0% |
| Math reasoning | MATH-500 | 500 | Final-answer accuracy, zero-shot; answers checked for mathematical equivalence | 45.0% |
| Code generation | MBPP (sanitized test split) | 257 | pass@1: share of problems whose generated code passes all tests | 70.0% |

Run these with the provided scripts and default settings (see the [README](README.md)). Staff also evaluate your checkpoint on held-out benchmarks for the same three skills; these are not released and are used only for [relative performance](#benchmark-credit).

## Training data and rules

Suggested datasets for SFT:

| Dataset | Split | Main use |
|---|---|---|
| [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | train | Math reasoning |
| [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | train (only split) | Instruction following and general SFT |
| [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | train (only split) | Code generation |

You may use subsets, filter them, or add other sources; document every source, split, sample count, and preprocessing step in your report.

**Rules** for all training data, including SFT, RL, and distillation data (violations result in a score of **zero for the entire project**):

1. Train only on the **training split** of any dataset, never on validation or test splits.
2. Never use examples from IFEval, MATH-500, or MBPP in training, including as prompts for generating distillation or RL data.
3. Check the data sources of any recipe you reuse. For example, the Tinker Cookbook math RL recipe mixes MATH test problems into its training data by default.

We review submitted code and check training data for overlap with evaluation data.

## The CPS 575 component

**CPS 575** teams complete **one** of the two options below, in addition to the SFT work above. It is worth 20% of the project grade.

### Option A: Extension beyond SFT (default)

| Approach | Examples | Counts as an extension? |
|---|---|---|
| SFT tuning | Learning rate, LoRA rank, batch size, training length | No |
| Data work | Mixture proportions, adding datasets, filtering, deduplication | No |
| Reinforcement learning | A reward-based training stage after SFT | Yes |
| Distillation | Training on teacher-generated targets, including on-policy distillation and self-distillation | Yes |
| Model merging | Combining separately trained checkpoints, such as task-specific models ([overview](https://cameronrwolfe.substack.com/p/model-merging)) | Yes |

Implement and evaluate at least one extension. Explain what you built and which parts come from existing libraries or tutorials. The extension does not need to improve scores to earn full credit, as long as it is correctly implemented and carefully evaluated.

For the extension, compare it against your own SFT model:

1. Train an SFT model (your **SFT baseline**).
2. Run the extension, then evaluate both models on all three public benchmarks with the same settings. Document differences in data, training budget, and any teacher model.
3. Explain gains, regressions, and trade-offs, separating what the experiment shows from what remains uncertain.

An ablation of a key design choice is encouraged but not required.

### Option B: Open-ended project (by proposal)

Teams interested in exploring tasks beyond the three listed here can propose their own project instead of Option A. Everything else stays the same: you still do the SFT work, and your final checkpoint is still graded on the three public benchmarks and the held-out benchmarks.

Submit a one-page proposal by **November 18, 11:59 PM** stating (1) the task and what you will do, (2) how you will evaluate it, and (3) the result you are aiming for. The instructor approves proposals; without approval, do Option A. An approved project is assessed on the ambition of the proposal and how close the results come to its target.

## Grading

The project is **20% of the course grade**. Percentages below are of the project grade.

| Component | **CPS 375** | **CPS 575** |
|---|---|---|
| Benchmark thresholds | 60% | 40% |
| Relative performance | 15% | 15% |
| Report and technical work | 25% | 45% (25% methodology, analysis, documentation + 20% CPS 575 component) |

### Benchmark credit

Both benchmark components use your **final checkpoint**: the single checkpoint you choose to be graded, usually your best one on the public benchmarks. It can come from SFT or from an extension, and it is the checkpoint named in the `submission.json` you upload.

- **Thresholds:** the three tasks are weighted equally. Credit on each task grows linearly from 0 at the untrained base model's score to full credit at the threshold:
  `credit = (your accuracy − base accuracy) / (threshold − base accuracy)`, clipped to [0, 1].
  Base-model scores: IFEval 47.5%, MATH-500 17.0%, MBPP 63.8%.
- **Relative performance:** ranked against other teams in the same course, using the held-out benchmarks. A model that generalizes, not one tuned to the public benchmarks, will do best.

### Report and technical work

- **CPS 375** **(25%):** training methodology, analysis of results, clarity, reproducibility, and required disclosures.
- **CPS 575** **(45%):** the same as **CPS 375** (25%), plus the CPS 575 component (20%). Option A is graded on correct implementation, actual execution, and a fair comparison against your SFT baseline; Option B on the ambition of the proposal and how close the results come to its target.

### AI / LLM policy

- You may use AI tools, including LLMs, for the coding and experimentation parts of the project.
- The report must be written by your team manually. Grammar and spell checkers are allowed; using AI to generate or rewrite report content is not.
- Disclose all AI use in the report's [AI usage disclosure](#2-report-pdf).
- If we suspect AI-written content in the report, your team may be called for an **oral exam**. The whole team will walk us through the project: what you did and why, your results, and how your code implements it, and answer our questions. In that case, the 25% report grade (methodology, analysis, documentation) is based on the oral exam instead of the submitted report.

## Submission

Submit three items on Gradescope.

### 1. `submission.json`

The full evaluation of your final checkpoint, produced by the [evaluation command](README.md#3-evaluate-the-final-checkpoint). Do not submit a `--limit` smoke-test run.

The checkpoint must be **published** and must **not expire**: staff rerun it to verify your results and to run the held-out benchmarks. `evaluation/train_and_publish.py` shows how to publish. Some Tinker Cookbook recipes save checkpoints that expire after 7 days (`ttl_seconds`); publish your final checkpoint yourself.

### 2. Report (PDF)

The main content is limited to **4 pages** for **CPS 375** and **5 pages** for **CPS 575**; the AI usage disclosure and Tinker feedback sections do not count toward the limit.

1. **Methodology:** datasets and splits, preprocessing, mixing strategy, training settings, number of experiments, and approximate cost. Identify your final checkpoint and explain why you chose it.
2. **Results and analysis:** scores on all three public benchmarks, comparisons, what worked and what did not, and limitations.
3. **AI usage disclosure:** for each AI tool you used, the model and version (e.g., Claude Opus 4.6), what you used it for (e.g., writing training code, debugging, generating training data), and how you used it.
4. **Tinker feedback:** what was difficult, whether the cookbook helped, what was missing, one suggested improvement, and whether you would use Tinker again.
5. **CPS 575** **only, the CPS 575 component:** the method, what you implemented, the results, and interpretation, including unsuccessful outcomes. For Option A, compare with your SFT baseline and include a table with both checkpoints' paths and their three public benchmark scores. For Option B, report the evaluation and target from your proposal.

### 3. Code (ZIP)

Training code, configurations, dependencies, and instructions sufficient to reproduce your final checkpoint. Identify any reused code. Do not include API keys.

**CPS 575** **only:** the code ZIP must also cover the CPS 575 component. For Option A, it must cover both the SFT baseline and the extension, and include both models' evaluation JSON files (e.g., `sft_results.json`, `extension_results.json`) and checkpoint paths. Both checkpoints must be published. For distillation, also identify the teacher model and how its targets were produced.
