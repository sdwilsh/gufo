"""One-line failure reports: check=<name> expected=<...> actual=<...>.

Every failed functional check prints exactly one such line on stderr, so a
truncated log still names the condition that failed and what was observed.
"""

import ast
from functools import lru_cache
from pathlib import Path
import traceback

LIMIT = 500
PREFIX = "check="


def one_line(value, limit=LIMIT):
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + f"...[+{len(text) - limit} chars]"


def check_line(name, expected, actual):
    return f"{PREFIX}{name} expected={one_line(expected)} actual={one_line(actual)}"


def causes(error):
    """The exception chain, outermost first, as 'Type: message' strings."""
    seen, chain = set(), []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__
    return chain


def describe(error):
    """str(error), plus the root cause that SDK wrappers (Connection error) hide."""
    chain = causes(error)
    return chain[0] if len(chain) == 1 else f"{chain[0]} [root cause: {chain[-1]}]"


@lru_cache(maxsize=None)
def _tree(path):
    return ast.parse(Path(path).read_text())


def _failed_assert(error):
    """(function, line, condition source) of the innermost assert in the traceback."""
    here = Path(__file__).resolve().parent
    found = None
    for frame, line in traceback.walk_tb(error.__traceback__):
        path = Path(frame.f_code.co_filename).resolve()
        if path.parent == here:
            found = (path, frame.f_code.co_name, line)
    if found is None:
        return None
    path, function, line = found
    best = None
    for node in ast.walk(_tree(str(path))):
        if isinstance(node, ast.Assert) and node.lineno <= line <= node.end_lineno:
            if best is None or node.lineno >= best.lineno:
                best = node
    condition = ast.unparse(best.test) if best is not None else "assertion holds"
    return function, line, condition


def failure_lines(error, last_check=None):
    """check= lines for a failed suite: its own lines, else derived from the error."""
    message = str(error)
    own = [line for line in message.splitlines() if line.startswith(PREFIX)]
    if own:
        return own
    name = last_check or "request"
    if isinstance(error, AssertionError):
        located = _failed_assert(error)
        if located is not None:
            function, line, condition = located
            return [check_line(f"{name}@{function}:{line}", condition,
                               f"false; {message or 'no message'}")]
    return [check_line(name, "the request completes", describe(error))]
