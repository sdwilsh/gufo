from billing.split import split_cents


class Invoice:
    def __init__(self, lines):
        self.lines = list(lines)  # (description, cents)

    @property
    def total(self):
        return sum(cents for _, cents in self.lines)

    def split_between(self, people):
        """Return {person: cents} sharing the invoice total."""
        shares = split_cents(self.total, len(people))
        return dict(zip(people, shares))
