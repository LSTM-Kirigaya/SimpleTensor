# -*- coding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.0
import numpy as np
from queue import Queue
from typing import Union

_default_graph = None

class Graph(object):  # our calculate graph
    def __init__(self):
        self.operations = []    # operation node
        self.Placeholders = []  # token node
        self.variables = []     # variable node

    def as_default(self):  # set the object as the global graph once it is instanced
        global _default_graph
        _default_graph = self


class Operation(object):  # operation object, it will generate all the operation object 
    def __init__(self, input_nodes : list = []):
        self.input_nodes = input_nodes
        self.consumers = []  # consumers log all the node that uses this operation
        self.output = None  
        for node in input_nodes:  # input nodes also have a consumers
            node.consumers.append(self)

        # add to global graph
        _default_graph.operations.append(self)
    
    def __add__(self, x):
        return add(self, x)
    
    def __str__(self):
        return f"Operation({str(self.output)})"
    
    def numpy(self):
        return np.array(self.output)

    def compute(self):  # work as a virtual function
        pass


class Placeholder(object):
    def __init__(self):
        self.consumers = []
        self.output = None
        # add to global graph
        _default_graph.Placeholders.append(self)
    
    def __add__(self, x):
        return add(self, x)

    def __str__(self):
        return f"Placeholder({str(self.output)})"
    
    def numpy(self):
        return np.array(self.output)

class Variable(object):
    def __init__(self, initial_value=None):
        if not isinstance(initial_value, np.ndarray):
            initial_value = np.array(initial_value)
        self.value = initial_value
        self.consumers = []
        self.output = None
        # add to global graph
        _default_graph.variables.append(self)
    
    def __add__(self, x):
        return add(self, x)
    
    def __str__(self):
        return f"Variable({str(self.output)})"
    
    def numpy(self):
        return np.array(self.output)


class add(Operation):
    def __init__(self, x : Union[Placeholder, Variable, Operation], y : Union[Placeholder, Variable, Operation]):
        super(add, self).__init__([x, y])

    def compute(self, x_value, y_value):
        # ensure the input node is an ndarray object
        if not isinstance(x_value, np.ndarray):
            x_value = np.array(x_value)
        if not isinstance(y_value, np.ndarray):
            y_value = np.array(y_value)
        return x_value + y_value


class negative(Operation):
    def __init__(self, x : Union[Placeholder, Variable, Operation]):
        super(negative, self).__init__([x])
    
    def compute(self, x_value):
        if not isinstance(x_value, np.ndarray):
            x_value = np.array(x_value)
        return -x_value

class matmul(Operation):  # matrix multiply object
    def __init__(self, x : Union[Placeholder, Variable, Operation], y : Union[Placeholder, Variable, Operation]):
        super(matmul, self).__init__([x, y])

    def compute(self, x_value, y_value):
        # ensure the input node is an ndarray object
        if not isinstance(x_value, np.ndarray):
            x_value = np.array(x_value)
        if not isinstance(y_value, np.ndarray):
            y_value = np.array(y_value)
        return x_value.dot(y_value)

class multiply(Operation):
    def __init__(self, x : Union[Placeholder, Variable, Operation], y : Union[Placeholder, Variable, Operation]):
        super(multiply, self).__init__([x, y])
    
    def compute(self, x_value, y_value):
        if not isinstance(x_value, np.ndarray):
            x_value = np.array(x_value)
        if not isinstance(y_value, np.ndarray):
            y_value = np.array(y_value)
        return x_value * y_value

class reduce_sum(Operation):
    def __init__(self, x : Union[Placeholder, Variable, Operation], axis=None):
        super(reduce_sum, self).__init__([x])
        self.axis = axis
    
    def compute(self, x_value):
        if not isinstance(x_value, np.ndarray):
            x_value = np.array(x_value)
        return np.sum(x_value, axis=self.axis)


class log(Operation):
    def __init__(self, x : Union[Placeholder, Variable, Operation]):
        super(log, self).__init__([x])
    
    def compute(self, x_value):
        if not isinstance(x_value, np.ndarray):
            x_value = np.array(x_value)
        return np.log(x_value)

 
class sigmoid(Operation):
    def __init__(self, x : Union[Placeholder, Variable, Operation]):
        super(sigmoid, self).__init__([x])
    
    def compute(self, x_value):
        return 1 / (1 + np.exp(-x_value))


class softmax(Operation):
    def __init__(self, x : Union[Placeholder, Variable, Operation], axis=0):
        super(softmax, self).__init__([x])
        self.axis = axis

    def compute(self, x_value):
        reduce_shape = list(np.array(x_value).shape)
        reduce_shape[self.axis] = 1
        return np.exp(x_value) / np.sum(np.exp(x_value), axis=self.axis).reshape(reduce_shape)   # reshape for boardcast


class Session(object):  # Session object calculate the input calculation graph
    def run(self, root_op : Operation, feed_dict={}):
        """
        operation: root node of the graph
        feed_dict: corresponding value of the Placeholder
        return: root node of the graph
        """
        all_nodes = self.__get_all_nodes(root_op)
        for node in all_nodes:
            if isinstance(node, Placeholder):   # if the node is a Placeholder, query the feed_dict to get the value
                node.output = np.array(feed_dict[node])
            elif isinstance(node, Variable):    # if the node is a Variable, its own value is the output
                node.output = np.array(node.value)
            else:       # if the node is an operation, call its 'compute' method to get the output
                node.inputs = [node.output for node in node.input_nodes]
                # use the 'compute' method to calculate operation node's output
                node.output = node.compute(*node.inputs)
                # transform to ndarray
                node.output = np.array(node.output)

        return root_op        # return root node
    
    
    def __get_all_nodes(self, operation):  # get all the nodes based on the root operation recursively
        all_nodes = []
        def recurse(node):
            if isinstance(node, Operation):   # only operation node has son node, and it needs recursion
                for n in node.input_nodes:
                    recurse(n)
            all_nodes.append(node)

        recurse(operation)   # call build-in recursive function
        return all_nodes


class GradientDescentOptimizer(object):
    def __init__(self, learning_rate : float = 1e-3):
        self.learning_rate = learning_rate
    
    def minimize(self, loss : Operation):
        learning_rate = self.learning_rate
        # operate its son class
        grad_table = compute_gradient(loss)
        # iterate all the nodes
        for node in grad_table:
            if isinstance(node, Variable):
                # find the corresponding grad of the node
                grad = grad_table[node]
                # use gradient descent
                node.value = - learning_rate * grad
        return 

class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}
    
    def register(self, target):
        def add_register_item(key, value):
            if not callable(value):
                raise Exception(f"register object must be callable! But receice:{value} is not callable!")
            if key in self._dict:
                print(f"warning: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
            self[key] = value
            return value

        if callable(target):            # 如果传入的目标可调用，说明之前没有给出注册名字，我们就以传入的函数或者类的名字作为注册名
            return add_register_item(target.__name__, target)
        else:                           # 如果不可调用，说明额外说明了注册的可调用对象的名字
            return lambda x : add_register_item(target, x)
    
    def __call__(self, target):
        return self.register(target)
    
    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]
    
    def __contains__(self, key):
        return key in self._dict
    
    def __str__(self):
        return str(self._dict)
    
    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()

# create register dict
_gradient_registry = Register()
@_gradient_registry("add")     # chain law
def __add_gradient(consumer : Operation, grad : float):
    return np.array([grad, grad])

@_gradient_registry("negative")
def __negative_gradient(consumer : Operation, grad : float):          
    return np.array([-1. * grad])                       

@_gradient_registry("reduce_sum")
def __reduce_sum_gradient(consumer : Operation, grad : float):
    return np.array([grad])

@_gradient_registry("matmul")
def __matmul_gradient(consumer : Operation, grad : float):
    A = np.array(consumer.input_nodes[0].output)
    B = np.array(consumer.input_nodes[1].output)
    grad = np.array(grad)
    return np.array([
        np.dot(grad, B.T),
        np.dot(A.T, grad)
    ])


@_gradient_registry("multiply")
def __multiply_gradient(consumer : Operation, grad : float):
    x = consumer.input_nodes[0].output
    y = consumer.input_nodes[1].output
    return np.array([y * grad, x * grad])

@_gradient_registry("log")
def __log_gradient(consumer : Operation, grad : float):
    x = consumer.input_nodes[0].output
    return np.array([1. / x * grad])

@_gradient_registry("sigmoid")
def __sigmoid_gradient(consumer : Operation, grad : float):
    x = consumer.input_nodes[0].output
    e_minus_x = np.exp(-1. * x)
    return np.array([(e_minus_x) / ((1 + e_minus_x) ** 2) * grad]) 

@_gradient_registry("softmax")
def __softmax_gradient(consumer : Operation, grad : float):
    x = consumer.input_nodes[0].output
    return x * (1 - x)



def compute_gradient(loss : Operation):
    grad_table = {}
    # initial value of loss
    grad_table[loss] = 1
    # use BFS to implement BP
    visit = set()
    queue = Queue()
    visit.add(loss)
    queue.put(loss)     # put first element into the queue

    while not queue.empty():
        node = queue.get()      # get head of queue

        if node != loss:
            grad_table[node] = 0
            for consumer in node.consumers:
                # get loss's gradient given to comsumer node
                lossgrad_wrt_consumer_output = grad_table[consumer]
                operation_name = consumer.__class__.__name__
                # bprop is the corresponding instance of operation node
                gradient_func = _gradient_registry[operation_name]
                # get all the gradient of all the concrete consumer nodes
                lossgrads_wrt_consumer_inputs = gradient_func(consumer, lossgrad_wrt_consumer_output)
                
                if len(consumer.input_nodes) == 1:
                    grad_table[node] += lossgrads_wrt_consumer_inputs
                else:
                    # consumer.input_nodes is a list, so the index is the method of list 
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)
                    grad_table[node] += lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]

        # put the node into the queue
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if input_node not in visit:
                    visit.add(input_node)
                    queue.put(input_node)
    return grad_table

Graph().as_default()
                
if __name__ == "__main__":
    a = Variable([[2, 1], [-1, -2]])
    b = Variable([1, 1])
    c = Placeholder()
    y = matmul(a, b)

    # TODO : nan appears in the graph, need to be solved
    # TODO : gradient need to be clipped
    # TODO : integate several loss functions