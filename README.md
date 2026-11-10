# CPS 375 / 575 Mini-Project: Multi-Task LLM Fine-Tuning

Fine-tune `Qwen/Qwen3.5-9B-Base` using Tinker to perform well on three tasks simultaneously: **Instruction Following (IFEval)**, **Math Reasoning (MATH-500)**, and **Code Generation (MBPP)**.

For the 375/575 requirements, grading breakdown, deliverables, and policies, see [`PROJECT.md`](PROJECT.md). This README covers installation and running the starter workflow.

## What the starter code provides

The starter script, [`evaluation/train_and_publish.py`](evaluation/train_and_publish.py), provides a minimal working training loop: creating a LoRA training client, preparing examples, updating parameters, and saving and publishing a checkpoint. It uses toy data. Your work is to connect real training data and implement your training strategy. Running the toy example alone does not fulfill the project requirements.

New to Tinker? Start with [`tinker_tutorial.ipynb`](tinker_tutorial.ipynb), a hands-on walkthrough of SFT and RL on Tinker. For reference, see the [Tinker documentation](https://tinker-docs.thinkingmachines.ai/) and the [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook), Thinking Machines' open-source library of training recipes (SFT, RL, distillation, and more).

## Setup and workflow

Run the commands below from the directory containing this handout and its `evaluation/` folder. In this repository, that directory is `student_handout/`.

### 1. Set up the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt

export TINKER_API_KEY="your-api-key-here"
```

Keep the virtual environment activated when running evaluations: the MBPP scorer executes generated code with the `python` on your `PATH`.

### 2. Check evaluation and toy training

```bash
# Quick evaluation check; not a final submission.
python -m evaluation.eval_all \
    --base_models Qwen/Qwen3.5-9B-Base --limit 5

# Minimal training example on toy data; saves and publishes a checkpoint.
python -m evaluation.train_and_publish
```

The training example writes checkpoint information to `evaluation/checkpoint_info.json`. Use its checkpoint path and matching base model when evaluating. Replace the toy data and implement your actual training approach before producing your final model. See the [suggested training datasets](PROJECT.md#training-data-and-rules) and [CPS 575 component](PROJECT.md#the-cps-575-component).

### 3. Evaluate the final checkpoint

```bash
# Omit --limit to evaluate all samples.
python -m evaluation.eval_all \
    --checkpoint_path "tinker://your-final-checkpoint" \
    --base_model Qwen/Qwen3.5-9B-Base \
    --temperature 0.0 --top_p 1.0 \
    --output_path evaluation/submission.json
```

The default generation settings are temperature 0.0 and top-p 1.0. If you change them, report the settings and use the same settings for your SFT-versus-extension comparison. The maximum generation length is fixed per task (1,024 tokens for IFEval and 4,096 for MATH-500 and MBPP) and is the same for all teams. A full evaluation takes a few minutes and costs about $0.60. Check that all three tasks completed and that their sample counts match the [benchmark table in PROJECT](PROJECT.md#benchmarks-and-thresholds).

For **CPS 575** Option A, evaluate the SFT and extension checkpoints separately, using distinct `--output_path` values such as `evaluation/sft_results.json` and `evaluation/extension_results.json`. Preserve these supporting files even if one of the checkpoints is also your final submission.

## Submit your results

The final evaluation command writes `evaluation/submission.json`. Make sure the checkpoint it names is published and does not expire (see [submission materials](PROJECT.md#1-submissionjson)), then upload that file to the Gradescope autograder assignment. Submit the report PDF and code ZIP according to the [submission requirements](PROJECT.md#submission), including the supporting SFT and extension results required for **CPS 575** Option A.

## File reference

| File | Purpose |
|---|---|
| [`PROJECT.md`](PROJECT.md) | Course-specific requirements, grading, deliverables, and policies |
| [`tinker_tutorial.ipynb`](tinker_tutorial.ipynb) | Hands-on Tinker tutorial: SFT and RL on a toy task |
| [`evaluation/train_and_publish.py`](evaluation/train_and_publish.py) | Toy SFT example; replace the data and adapt the training approach |
| [`evaluation/eval_all.py`](evaluation/eval_all.py) | Evaluate base models or a checkpoint on all three benchmarks |
| [`evaluation/eval_ifeval.py`](evaluation/eval_ifeval.py) | IFEval evaluation |
| [`evaluation/eval_math.py`](evaluation/eval_math.py) | MATH-500 evaluation, zero-shot |
| [`evaluation/math_scorer.py`](evaluation/math_scorer.py) | Answer-equivalence scorer used for MATH-500 |
| [`evaluation/eval_code.py`](evaluation/eval_code.py) | MBPP evaluation with local code execution |
| [`evaluation/run_eval.sh`](evaluation/run_eval.sh) | Shortcut for evaluating a checkpoint |
| [`requirements.txt`](requirements.txt) | Pinned dependencies |

You can also run individual tasks from the handout directory:

```bash
python -m evaluation.eval_ifeval --limit 5
python -m evaluation.eval_math --limit 5
python -m evaluation.eval_code --limit 5
```
