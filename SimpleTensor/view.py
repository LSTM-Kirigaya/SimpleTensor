from SimpleTensor.core     import Operation, Placeholder, Variable, Node
from SimpleTensor.constant import runtime
from collections import deque, defaultdict
import numpy as np

def get_node_attr(node: Node) -> dict:
    if isinstance(node, Placeholder):
        node_attr = {"color": "#e5864d", "style" : "filled", "fontcolor": "white", "shape" : "rectangle"}
    elif isinstance(node, Variable):
        node_attr = {"color": "#556fca", "style" : "filled", "fontcolor": "white"}
    elif isinstance(node, Operation):
        node_attr = {"color": "#4dcb88", "style" : "filled", "fontcolor": "white", "shape" : "rectangle"}
    
    return node_attr

def get_node_label(node: Node, show_grad: bool, show_lines: int) -> str:
    name = node.__class__.__name__
    if isinstance(node, Placeholder):
        return name
    if show_grad is False or runtime.grad_table is None or node not in runtime.grad_table:
        return name
    else:
        grad = runtime.grad_table[node]
        if isinstance(grad, np.ndarray) and grad.shape[0] > show_lines:
            grad = grad[:show_lines]
        return "{}\n{}".format(name, grad)


def view_graph(file_name : str = "./dot", format:str ="png", direction: str ="UD", show_grad: bool=False, show_detail: bool=True,
               show_lines: int=20) -> None:
    """
        generate the graph of the whole calculation graph
    """
    from graphviz import Digraph
    dot = Digraph('G', format=format)

    node = runtime.global_calc_graph[-1]

    if not isinstance(node, Operation):
        raise ValueError("input node must be an Operation!")
    queue = deque()
    visit = set()
    
    queue.append(node)
    visit.add(node)
    node2id = {}

    for index, n in enumerate(runtime.global_calc_graph):
        if n.node_name:
            # notice that the subgraph in graphviz must be named with 'cluster'
            with dot.subgraph(name="cluster_{}".format(n.node_name)) as c:
                c.attr(style='filled', color='lightgrey')
                c.node(
                    name=str(index), 
                    label=get_node_label(n, show_grad, show_lines=show_lines), 
                    **get_node_attr(n)
                )
                c.attr(label=n.node_name)

        else:
            dot.node(
                name=str(index), 
                label=get_node_label(n, show_grad, show_lines=show_lines), 
                **get_node_attr(n)
            )
        node2id[n] = str(index)
            

    while len(queue) > 0:
        cur_node = queue.popleft()
        for input_node in cur_node.input_nodes:
            if isinstance(input_node, Operation) and input_node not in visit:   
                visit.add(input_node)
                queue.append(input_node)

            dot.edge(node2id[input_node], node2id[cur_node], color="#31a3e5")

    dot.attr(rankdir=direction)
    dot.render(filename=file_name, cleanup=True)