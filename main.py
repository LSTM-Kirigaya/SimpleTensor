from SimpleTensor.core import Placeholder
from SimpleTensor.core import Variable
from SimpleTensor.core import matmul, softmax, log, negative, sigmoid, add, reduce_sum, multiply
from SimpleTensor.core import Session
from SimpleTensor.core import GradientDescentOptimizer

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(120)
# samples
red_points = np.random.randn(50, 2) - 2
blue_points = np.random.randn(50, 2) + 2

X = Placeholder()
W = Variable(np.random.randn(2, 2))
b = Variable(np.random.randn(1, 2))
c = Placeholder()

f = matmul(X, W) + b        # linear layer
p = softmax(f)

loss = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))
optimizer = GradientDescentOptimizer(learning_rate=1e-3)

# define feed dict
feed_dict = {
    X : np.concatenate([blue_points, red_points], axis=0),
    c : [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)
}

session = Session()

for epoch in range(10):
    loss = session.run(root_op=loss, feed_dict=feed_dict)
    optimizer.minimize(loss)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss:{loss.numpy()}")

plt.style.use("seaborn")
plt.scatter(red_points[:, 0], red_points[:, 1], color="red")
plt.scatter(blue_points[:, 0], blue_points[:, 1], color="blue")

plt.show()
