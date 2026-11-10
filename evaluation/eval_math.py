"""
Evaluate a checkpoint on MATH-500 (competition math, final-answer accuracy).

Uses Inspect AI's built-in MATH task (zero-shot, answer on a final "ANSWER:" line), restricted to the
500 problems of MATH-500, and scores answers with math_scorer.math_answer_match.

Usage:
    python -m evaluation.eval_math
    python -m evaluation.eval_math --checkpoint_path "tinker://..."
    python -m evaluation.eval_math --limit 20
"""

import argparse
import asyncio
import json
import logging
import os

import tinker
from datasets import load_dataset
from inspect_ai import eval_async, task_with
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import CORRECT
from inspect_evals.math.math import math

from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from tinker_cookbook.model_info import get_recommended_renderer_name

from evaluation.math_scorer import math_answer_match

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "Qwen/Qwen3.5-9B-Base"


def math500_task(config):
    """Inspect's MATH task restricted to the 500 MATH-500 problems, with our answer scorer."""
    keep = set(load_dataset("HuggingFaceH4/MATH-500", split="test")["problem"])
    task = math(shuffle=False)
    dataset = MemoryDataset([s for s in task.dataset if s.input in keep], name="math500")
    # The stock task sets temperature=0.5 at task level, which would override the model config.
    return task_with(task, dataset=dataset, scorer=[math_answer_match()], config=config)


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

    log_dir = args.log_dir or os.path.join(EVAL_DIR, "inspect-logs")
    results = await eval_async(
        tasks=[math500_task(config)],
        model=[model],
        limit=args.limit,
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
                metrics[f"math500/{name}"] = score.value

    # Extract per-sample results from the log file
    samples = []
    for r in results:
        if r.location:
            log = read_eval_log(r.location)
            if log.samples:
                for s in log.samples:
                    correct = any(score.value == CORRECT for score in (s.scores or {}).values())
                    samples.append({"id": s.id, "correct": correct})

    logger.info(f"MATH-500 results: {json.dumps(metrics, indent=2)}")
    return {"metrics": metrics, "samples": samples}


def main():
    p = argparse.ArgumentParser(description="Evaluate on MATH-500")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--base_model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--renderer_name", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="Max samples (None=all 500)")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    result = asyncio.run(run(p.parse_args()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
