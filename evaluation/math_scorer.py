"""MATH answer scorer: correct if math_verify, sympy, or normalized-string equivalence says so.

The stock sympy scorer marks any answer it can't parse as LaTeX (tuples, intervals, unicode pi)
as wrong, even when it is identical to the target; on our checkpoints that undercounts by 3-5 points.
"""

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_evals.math.utils import extract_answer, is_equiv, is_equiv_sympy, normalize_final_answer
from math_verify import parse, verify


@scorer(metrics=[accuracy(), stderr()])
def math_answer_match():
    """Score the last 'ANSWER:' line against the target with three equivalence checks."""
    async def score(state: TaskState, target: Target) -> Score:
        answer = extract_answer(state.output.completion)
        if not answer:
            return Score(value=INCORRECT, answer="", explanation="no ANSWER line")
        try:
            mv = bool(verify(parse(f"${target.text}$"), parse(f"${answer}$")))
        except Exception:
            mv = False
        correct = (mv or await is_equiv(answer, target.text)
                   or await is_equiv_sympy(await normalize_final_answer(answer),
                                           await normalize_final_answer(target.text)))
        return Score(value=CORRECT if correct else INCORRECT, answer=answer,
                     metadata={"math_verify": mv})
    return score
