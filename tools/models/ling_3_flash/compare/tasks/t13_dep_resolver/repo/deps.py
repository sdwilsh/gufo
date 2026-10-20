class CycleError(ValueError):
    pass


def resolve(graph):
    order = []

    def visit(name):
        for dep in sorted(graph.get(name, [])):
            visit(dep)
        order.append(name)

    for name in sorted(graph):
        visit(name)
    return order
