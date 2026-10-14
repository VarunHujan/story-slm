"""
=============================================================================
Story-SLM2: Deterministic Utility Tools (utility_tools.py)
=============================================================================
Small, from-scratch language models cannot reliably recall precise facts
(exact dates) or compute exact arithmetic -- this was verified directly:
asked "what is today's date?" with the real date injected into context, the
model echoed the correct month and day but hallucinated the wrong year. No
amount of training fixes this reliably; frozen weights have no clock, and
next-token prediction isn't how exact digit-by-digit arithmetic works.

The fix used by every real production system: detect these specific
question types with plain code and answer them deterministically, bypassing
the neural network entirely for the parts where it can't be trusted. The
model is only used for what it's actually good at -- generating the story
text itself.

These checks are deliberately narrow (only match confidently): a missed
match just falls through to normal story generation, which is the safe
default.
=============================================================================
"""

import ast
import datetime
import operator
import random
import re
from typing import Optional

# The underlying FACT (the real date/time/calculation) must always be
# identical for equivalent questions -- "what day is today" and "what is
# today's date" are the same question and must get the same true answer.
# What should vary is the WORDING, so repeated identical phrasing doesn't
# read as an obviously hardcoded string. Accuracy never changes; only style.
DATE_PHRASINGS = [
    "Today is {weekday}, {date}.",
    "It's {weekday}, {date} today.",
    "Today's date is {weekday}, {date}.",
    "The date today is {weekday}, {date}.",
]
TIME_PHRASINGS = [
    "The current time is {time}.",
    "It's {time} right now.",
    "Right now it's {time}.",
]
MATH_PHRASINGS = [
    "The result of {expr} is {result}.",
    "{expr} equals {result}.",
    "That comes out to {result}.",
    "{expr} = {result}.",
]


DATE_PATTERNS = [
    r"\bwhat'?s?\s+(?:is\s+)?(?:today'?s?\s+)?date\b",
    r"\bwhat\s+day\s+is\s+(it|today|it\s+today)\b",
    r"\btoday'?s?\s+date\b",
    r"\bcurrent\s+date\b",
    r"\bwhat\s+is\s+the\s+date\s+today\b",
    r"\bwhat\s+day\s+of\s+the\s+week\b",
]

TIME_PATTERNS = [
    r"\bwhat\s+time\s+is\s+it\b",
    r"\bcurrent\s+time\b",
    r"\bwhat'?s?\s+the\s+time\b",
]

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node):
    """Evaluates only numeric literals and +-*/** combined via a real AST,
    not Python's eval() -- no arbitrary code execution is possible since
    only this fixed, whitelisted set of node types is ever interpreted."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError("unsupported expression")


def try_answer_date_time(prompt: str) -> Optional[str]:
    lower = prompt.lower()
    if any(re.search(p, lower) for p in DATE_PATTERNS):
        today = datetime.date.today()
        return random.choice(DATE_PHRASINGS).format(
            weekday=today.strftime("%A"), date=today.strftime("%B %d, %Y")
        )
    if any(re.search(p, lower) for p in TIME_PATTERNS):
        return random.choice(TIME_PHRASINGS).format(time=datetime.datetime.now().strftime("%I:%M %p"))
    return None


_MATH_EXPR_RE = re.compile(r"^[\s\d\.\+\-\*/\(\)\^]+$")


def try_answer_math(prompt: str) -> Optional[str]:
    # Strip common leading phrasing so "what is 2+5" and "calculate 2+5" both
    # reduce to the bare expression "2+5" before validating/evaluating it.
    stripped = re.sub(
        r"^\s*(what\s+is|calculate|compute|solve|what'?s)\s*:?\s*",
        "", prompt.strip(), flags=re.IGNORECASE,
    ).rstrip("?").strip()

    if not stripped or not _MATH_EXPR_RE.match(stripped):
        return None
    if not any(ch.isdigit() for ch in stripped):
        return None  # reject bare "()" or whitespace-only matches

    try:
        expr = stripped.replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree.body)
    except Exception:
        return None  # not a valid expression -- fall through to the model

    if isinstance(result, float):
        if result.is_integer():
            result = int(result)
        else:
            result = round(result, 6)
    return random.choice(MATH_PHRASINGS).format(expr=stripped, result=result)


def try_answer_deterministically(prompt: str) -> Optional[str]:
    """Checked before generation. Returns a direct answer string if the
    prompt matches a known deterministic question type, else None (meaning:
    fall through to normal model generation)."""
    return try_answer_date_time(prompt) or try_answer_math(prompt)
