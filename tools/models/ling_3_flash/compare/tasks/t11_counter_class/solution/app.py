from counter import Counter


class Server:
    def __init__(self, counter=None):
        self.counter = counter if counter is not None else Counter()

    def handle(self, path):
        self.counter.hit(path)
        return f"ok {path}"

    def stats(self):
        return self.counter.total()
