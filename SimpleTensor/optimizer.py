import abc
from SimpleTensor import Operation, Variable
from SimpleTensor import runtime
from queue import Queue
# TODO : use deque in collections instead


# optimizer
class Optimizer(abc.ABC):
    def __init__(self, learning_rate : float = 1e-3):
        """
            base for all the optimizer
        """
        self.learning_rate = learning_rate
    
    def __backwards(self, op_node : Operation):
        """
            do the BP from the op_node, 
            return a gradient dict including op_node's gradients with respect to all the nodes before op_node
        """
        # wo will do the BP through BFS
        grad_table = {}
        grad_table[op_node] = 1.
        visit_nodes = set()
        queue = Queue()
        visit_nodes.add(op_node)
        queue.put(op_node)

        while not queue.empty():
            cur_node = queue.get()

            if cur_node != op_node:
                grad_table[cur_node] = 0.
                for next_node in cur_node.next_nodes:
                    grad_loss_wrt_next_node = grad_table[next_node]                                 # loss gradient of next_node
                    next_node_op_name = next_node.__class__.__name__                                # next_node must be an Operation, we get its name
                    gradient_func = runtime.gradient_func[next_node_op_name]                     # get next_node's corresponding gradient function

                    grad_loss_wrt_cur_node = gradient_func(next_node, grad_loss_wrt_next_node)      # call the gradient function to get the sub-gradient
                    
                    if len(next_node.input_nodes) == 1:                                            # if next_node represents monocular operators, then add to total gradient directly
                        grad_table[cur_node] += grad_loss_wrt_cur_node[0]
                    else:                                                                           # else get the portion size of gradient
                        cur_node_in_next_node_index = next_node.input_nodes.index(cur_node)
                        grad_table[cur_node] += grad_loss_wrt_cur_node[cur_node_in_next_node_index]

            if isinstance(cur_node, Operation):                                                     # put next op node into queue to do the BFS
                for input_node in cur_node.input_nodes:
                    if input_node not in visit_nodes:                                               # only add nodes which haven't been updated/visited yet
                        visit_nodes.add(input_node)
                        queue.put(input_node)

        return grad_table

    @abc.abstractmethod
    def minimize(self, loss_node : Operation):
        """
            concrete optimizer method, 
            this method will update parameters before "loss" node(include loss)
        """
        pass


class SGD(Optimizer):   # Stochastic gradient descent 
    def __init__(self, learning_rate : float = 1e-3):
        super().__init__(learning_rate=learning_rate)
    
    def minimize(self, loss_node : Operation):
        lr = self.learning_rate
        grad_table = self._Optimizer__backwards(op_node=loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                node.data -= lr * grad

        runtime.grad_table = grad_table
        return grad_table

# TODO : More powerful optimizer
# https://www.jianshu.com/p/aebcaf8af76e