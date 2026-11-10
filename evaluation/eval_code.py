"""
Evaluate a checkpoint on MBPP (Python code generation, pass@1).

Uses Inspect AI's built-in MBPP task (sanitized test split, 257 problems). Generated code is executed
locally against each problem's assert-based tests.

Usage:
    python -m evaluation.eval_code
    python -m evaluation.eval_code --checkpoint_path "tinker://..."
    python -m evaluation.eval_code --limit 20
"""

import argparse
import asyncio
import json
import logging
import os

import tinker
from inspect_ai import Epochs, eval_async, task_with
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import CORRECT
from inspect_evals.mbpp import mbpp

from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from tinker_cookbook.model_info import get_recommended_renderer_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "Qwen/Qwen3.5-9B-Base"


async def run(args):
    renderer_name = args.renderer_name or get_recommended_renderer_name(args.base_model)
    logger.info(f"Model: {args.base_model} | Renderer: {renderer_name}")

    sc = tinker.ServiceClient()
    if args.checkpoint_path:
        sampling_client = sc.create_sampling_client(model_path=args.checkpoint_path).result()
    else:
        sampling_client = sc.create_sampling_client(base_model=args.base_model).result()

    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=args.base_model,
        sampling_client=sampling_client,
        verbose=args.verbose,
    )
    config = GenerateConfig(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    model = Model(api=api, config=config)

    # The stock task samples 5 times at temperature 0.5; we score one generation with the given settings.
    task = task_with(mbpp(), config=config, epochs=Epochs(1, "mean"))

    log_dir = args.log_dir or os.path.join(EVAL_DIR, "inspect-logs")
    results = await eval_async(
        tasks=[task],
        model=[model],
        limit=args.limit,
        sandbox="local",
        debug_errors=True,
        retry_on_error=5,
        fail_on_error=False,
        log_dir=log_dir,
        max_connections=512,
    )

    # Extract aggregate metrics
    metrics = {}
    for r in results:
        if r.results and r.results.scores:
            for name, score in r.results.scores[0].metrics.items():
                metrics[f"mbpp/{name}"] = score.value

    # Extract per-sample results from the log file
    samples = []
    for r in results:
        if r.location:
            log = read_eval_log(r.location)
            if log.samples:
                for s in log.samples:
                    correct = any(score.value == CORRECT for score in (s.scores or {}).values())
                    samples.append({"id": s.id, "correct": correct})

    logger.info(f"MBPP results: {json.dumps(metrics, indent=2)}")
    return {"metrics": metrics, "samples": samples}


def main():
    p = argparse.ArgumentParser(description="Evaluate on MBPP")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--base_model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--renderer_name", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="Max samples (None=all 257)")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    result = asyncio.run(run(p.parse_args()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
