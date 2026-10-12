class CycleError(ValueError):
    pass


def resolve(graph):
    nodes = set(graph)
    for deps in graph.values():
        nodes.update(deps)
    remaining = {n: set(graph.get(n, ())) for n in nodes}
    order = []
    while remaining:
        ready = sorted(n for n, deps in remaining.items() if not deps)
        if not ready:
            raise CycleError("dependency cycle among: " + ", ".join(sorted(remaining)))
        node = ready[0]
        order.append(node)
        del remaining[node]
        for deps in remaining.values():
            deps.discard(node)
    return order
