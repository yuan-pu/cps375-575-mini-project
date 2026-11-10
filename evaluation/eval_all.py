"""
Evaluate base models or trained checkpoints on the three core tasks (IFEval, MATH-500, MBPP).

Usage (from the handout directory):
    # Evaluate one or more base models:
    python -m evaluation.eval_all --base_models Qwen/Qwen3.5-9B-Base

    # Evaluate a trained checkpoint and write evaluation/submission.json:
    python -m evaluation.eval_all --checkpoint_path "tinker://..." --base_model Qwen/Qwen3.5-9B-Base

    # Quick smoke test (5 samples per task):
    python -m evaluation.eval_all --base_models Qwen/Qwen3.5-9B-Base --limit 5
"""

import argparse
import asyncio
import json
import logging
import os

from tinker_cookbook.model_info import get_recommended_renderer_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
# Fixed per task so that all teams are graded under the same generation budget.
MAX_TOKENS = {"ifeval": 1024, "math500": 4096, "mbpp": 4096}

async def run_core(base_model, checkpoint_path, renderer_name, temperature,
                   top_p, limit, log_dir, verbose):
    """Run core 3 tasks for a single model/checkpoint. Returns (all_metrics, task_results)."""
    if not renderer_name:
        renderer_name = get_recommended_renderer_name(base_model)
    label = checkpoint_path or f"base:{base_model}"
    logger.info(f"Evaluating: {label}")
    logger.info(f"Renderer: {renderer_name} | temperature={temperature} top_p={top_p}")

    all_metrics = {}
    task_results = {}

    task_args = dict(
        checkpoint_path=checkpoint_path, base_model=base_model,
        renderer_name=renderer_name, temperature=temperature, top_p=top_p,
        limit=limit, log_dir=log_dir, verbose=verbose,
    )

    # 1. IFEval
    logger.info("=" * 60)
    logger.info("TASK 1/3: IFEval (Instruction Following)")
    logger.info("=" * 60)
    try:
        from evaluation.eval_ifeval import run as run_ifeval
        result = await run_ifeval(argparse.Namespace(**task_args, max_tokens=MAX_TOKENS["ifeval"]))
        all_metrics.update(result["metrics"])
        task_results["ifeval"] = result
    except Exception as e:
        logger.error(f"IFEval failed: {e}")
        all_metrics["ifeval/error"] = str(e)
        task_results["ifeval"] = {"error": str(e)}

    # 2. MATH-500
    logger.info("=" * 60)
    logger.info("TASK 2/3: MATH-500 (Math Reasoning)")
    logger.info("=" * 60)
    try:
        from evaluation.eval_math import run as run_math
        result = await run_math(argparse.Namespace(**task_args, max_tokens=MAX_TOKENS["math500"]))
        all_metrics.update(result["metrics"])
        task_results["math500"] = result
    except Exception as e:
        logger.error(f"MATH-500 failed: {e}")
        all_metrics["math500/error"] = str(e)
        task_results["math500"] = {"error": str(e)}

    # 3. MBPP
    logger.info("=" * 60)
    logger.info("TASK 3/3: MBPP (Code Generation)")
    logger.info("=" * 60)
    try:
        from evaluation.eval_code import run as run_code
        result = await run_code(argparse.Namespace(**task_args, max_tokens=MAX_TOKENS["mbpp"]))
        all_metrics.update(result["metrics"])
        task_results["mbpp"] = result
    except Exception as e:
        logger.error(f"MBPP failed: {e}")
        all_metrics["mbpp/error"] = str(e)
        task_results["mbpp"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY (core): {label}")
    print("=" * 60)
    print(json.dumps(all_metrics, indent=2))
    print("=" * 60)

    return all_metrics, task_results


def print_comparison(all_data, title):
    """Print a comparison table for a dict of {model: metrics}."""
    if len(all_data) < 2:
        return
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    all_keys = set()
    for m in all_data.values():
        all_keys.update(k for k in m if "error" not in k and "stderr" not in k)
    models_list = list(all_data.keys())
    header = f"{'Metric':<45}" + "".join(f"{m.split('/')[-1]:>15}" for m in models_list)
    print(header)
    print("-" * len(header))
    for key in sorted(all_keys):
        row = f"  {key:<43}"
        for model in models_list:
            val = all_data[model].get(key, "N/A")
            if isinstance(val, (int, float)):
                row += f"{val:>15.4f}"
            else:
                row += f"{str(val):>15}"
        print(row)
    print("=" * 80)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Evaluate on IFEval, MATH-500, and MBPP")
    p.add_argument("--base_models", type=str, nargs="+", default=None,
                    help="One or more base models to evaluate (e.g. Qwen/Qwen3.5-9B-Base)")
    p.add_argument("--checkpoint_path", type=str, default=None,
                    help="Tinker checkpoint path to evaluate")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-9B-Base",
                    help="Base model for the checkpoint (used for tokenizer/renderer)")
    p.add_argument("--renderer_name", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (default: 0.0 = greedy)")
    p.add_argument("--top_p", type=float, default=1.0,
                    help="Top-p (nucleus) sampling (default: 1.0 = disabled)")
    p.add_argument("--limit", type=int, default=None, help="Max samples per task")
    p.add_argument("--output_path", type=str, default=None,
                    help="Path for submission JSON (default: evaluation/submission.json)")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not args.base_models and not args.checkpoint_path:
        p.error("Provide either --base_models or --checkpoint_path")

    if args.checkpoint_path:
        # Checkpoint mode: evaluate one checkpoint, save submission.json
        metrics, task_results = asyncio.run(run_core(
            base_model=args.base_model,
            checkpoint_path=args.checkpoint_path,
            renderer_name=args.renderer_name,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
            log_dir=args.log_dir,
            verbose=args.verbose,
        ))
        submission = {
            "checkpoint_path": args.checkpoint_path,
            "base_model": args.base_model,
            "settings": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": MAX_TOKENS,
                "limit": args.limit,
            },
            **task_results,
        }
        out_path = args.output_path or os.path.join(EVAL_DIR, "submission.json")
        save_json(out_path, submission)
        logger.info(f"Submission saved to {out_path}")

    else:
        # Baseline mode: evaluate one or more base models
        core_path = os.path.join(EVAL_DIR, "baseline_results.json")

        core_baselines = load_json(core_path).get("models", {})

        for model in args.base_models:
            # Core tasks
            if model in core_baselines and args.limit is None:
                logger.info(f"SKIP core: {model} already in baseline_results.json")
            else:
                print("\n" + "#" * 60)
                print(f"# BASELINE (core): {model}")
                print("#" * 60)
                metrics, _ = asyncio.run(run_core(
                    base_model=model,
                    checkpoint_path=None,
                    renderer_name=None,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    limit=args.limit,
                    log_dir=args.log_dir,
                    verbose=args.verbose,
                ))
                if args.limit is None:
                    core_baselines[model] = metrics
                    save_json(core_path, {"type": "baseline", "models": core_baselines})
                    logger.info(f"Core baselines updated: {core_path}")
                else:
                    logger.info(f"--limit={args.limit} set, results NOT saved")

        # Print comparison table
        print_comparison(core_baselines, "BASELINE COMPARISON")


if __name__ == "__main__":
    main()
