import abc
from SimpleTensor import Operation, Variable, Placeholder, Data
from SimpleTensor import runtime
from collections import deque
import numpy as np
from SimpleTensor.constant import EPSILON

def backwards(op_node : Operation):
    """
        do the BP from the op_node, 
        return a gradient dict including op_node's gradients with respect to all the nodes before op_node
    """
    # we will do the BP through BFS
    grad_table = {}
    grad_table[op_node] = 1.
    visit_nodes = set()
    queue = deque()
    visit_nodes.add(op_node)
    queue.append(op_node)

    while len(queue) > 0:
        cur_node = queue.popleft()
        # TODO: try to do loop
        if cur_node != op_node and not (isinstance(cur_node, Placeholder) or isinstance(cur_node, Data)):
            grad_table[cur_node] = 0
            for next_node in cur_node.next_nodes:
                grad_loss_wrt_next_node : np.ndarray = grad_table[next_node]                                 # loss gradient of next_node
                next_node_op_name : str = next_node.__class__.__name__                                # next_node must be an Operation, we get its name
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
                    queue.append(input_node)

    return grad_table


# optimizer
class Optimizer(abc.ABC):
    def __init__(self, learning_rate : float = 1e-3):
        """
            base for all the optimizer
        """
        self.learning_rate = learning_rate    

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
        grad_table = backwards(op_node=loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                node.data -= lr * grad

        runtime.grad_table = grad_table
        return grad_table

# TODO : More powerful optimizer
# https://www.jianshu.com/p/aebcaf8af76e
class Momentum(Optimizer):
    def __init__(self, learning_rate: float = 0.001, gamma=0.7):
        super().__init__(learning_rate)
        # save gradient of each node
        self.node2v = {}
        self.gamma = gamma
    
    def minimize(self, loss_node: Operation):
        lr = self.learning_rate
        grad_table = self._Optimizer__backwards(op_node=loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                if node not in self.node2v:
                    self.node2v[node] = lr * grad
                else:
                    self.node2v[node] = self.gamma * self.node2v[node] + lr * grad
                node.data -= self.node2v[node]

        runtime.grad_table = grad_table
        return grad_table
    
class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float = 0.001):
        super().__init__(learning_rate)
        self.node2sumgrad = {}
    
    def minimize(self, loss_node : Operation):
        lr = self.learning_rate
        grad_table = self._Optimizer__backwards(op_node=loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                if node not in self.node2sumgrad:
                    self.node2sumgrad[node] = grad * grad
                else:
                    self.node2sumgrad[node] += grad * grad
                
                node.data -= lr * grad / np.sqrt(self.node2sumgrad[node] + EPSILON)

        runtime.grad_table = grad_table
        return grad_table

class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 0.001, gamma=0.7):
        super().__init__(learning_rate)
        self.gamma = gamma
        self.node2v = {}
    
    def minimize(self, loss_node : Operation):
        lr = self.learning_rate
        grad_table = self._Optimizer__backwards(op_node=loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                if node not in self.node2v:
                    self.node2v[node] = (1 - self.gamma) * grad * grad
                else:
                    self.node2v[node] = self.gamma * self.node2v[node] + (1 - self.gamma) * grad * grad

                node.data -= lr * grad / (np.sqrt(self.node2v[node] + EPSILON))

        runtime.grad_table = grad_table
        return grad_table

class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1 = 0.9, beta2 = 0.999):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2

        self.prod_beta1 = 1
        self.prod_beta2 = 1
        self.node2m = {}            # history value
        self.node2v = {}            # accumulate value
    
    def minimize(self, loss_node : Operation):
        lr = self.learning_rate
        grad_table = self._Optimizer__backwards(op_node=loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                if node not in self.node2m:
                    self.node2m[node] = (1 - self.beta1) * grad
                else:
                    self.node2m[node] = self.beta1 * self.node2m[node] + (1 - self.beta1) * grad
                
                if node not in self.node2v:
                    self.node2v[node] = (1 - self.beta2) * grad * grad
                else:
                    self.node2v[node] = self.beta2 * self.node2v[node] + (1 - self.beta2) * grad * grad
                
                self.prod_beta1 *= self.beta1
                self.prod_beta2 *= self.beta2
                m_hat = self.node2m[node] / (1 - self.prod_beta1)
                v_hat = self.node2v[node] / (1 - self.prod_beta2)

                node.data -= lr * m_hat / (np.sqrt(v_hat + EPSILON))

        runtime.grad_table = grad_table
        return grad_table
    

optimizers = {
    "SGD" : SGD,
    "Momentum" : Momentum,
    "AdaGrad" : AdaGrad,
    "RMSProp" : RMSProp,
    "Adan" : Adam
}