from graphviz import Digraph


def topological_sorting(root):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            
            for child in v._prev:
                build_topo(child)
            
            topo.append(v) # Each node is appended only if their children has being processed

    build_topo(root)
    return topo


def trace(root):
    # Building a set of all nodes and edges in the graph
    nodes, edges = set(), set()

    # Adding recursivly the previous `Values`` of v
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root): # expects expression of Values
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"}) # LR: left to right
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n)) # the specifier/name of each node (in memory)

        # Creating a recragular `record` for every node of the graph
        if n.label:
            dot.node(name=uid, label="{%s | d:%.4f | g:%.4f}" % (n.label, n.data, n.grad), shape="record")
        else:
            dot.node(name=uid, label="{d:%.4f | g:%.4f}" % (n.data, n.grad), shape="record")

        # Creating a node for the operation if this node is produced by one
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid) # connect the 2 nodes: (uid + n._op) --> (uid)

    # Connecting the `other` Value to the operator of `self`
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
