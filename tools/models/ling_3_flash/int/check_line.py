"""The failed-check line shared by ling-int-suite and ling-serve-suite."""
import sys


def check_line(name, expected, actual):
    """One failed check as `check=NAME expected=... actual=...` (same shape as
    tests/functional/check_report.py), on stderr."""
    def one(value, limit=500):
        text = " ".join(str(value).split())
        return text if len(text) <= limit else text[:limit] + "...[+%d chars]" % (len(text) - limit)
    print("check=%s expected=%s actual=%s" % (name, one(expected), one(actual)), file=sys.stderr, flush=True)
