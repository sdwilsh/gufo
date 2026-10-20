import counter


class Server:
    def handle(self, path):
        counter.hit(path)
        return f"ok {path}"

    def stats(self):
        return counter.total()
