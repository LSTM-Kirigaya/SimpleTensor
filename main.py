from SimpleTensor.core import Placeholder
from SimpleTensor.core import Variable
from SimpleTensor.core import matmul, softmax, log, negative, sigmoid, add, reduce_sum, multiply
from SimpleTensor.core import cross_entropy
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
p = softmax(f, axis=1)

loss = cross_entropy(predict=p, label=c, one_hot=False)
optimizer = GradientDescentOptimizer(learning_rate=1e-3)

# define feed dict
feed_dict = {
    X : np.concatenate([blue_points, red_points], axis=0),
    c : [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)
}

session = Session()

losses = []

for epoch in range(10):
    loss = session.run(root_op=loss, feed_dict=feed_dict)
    losses.append(loss.numpy)
    optimizer.minimize(loss)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss:{loss.numpy}")

W_weight = W.to_numpy()
b_weight = b.numpy

x = np.arange(-4, 4, 0.1)
y = -W_weight[0][0] / W_weight[1][0] * x - b_weight[0][0] / W_weight[1][0]


plt.style.use(['dark_background'])
plt.subplot(1, 2, 1)
plt.plot(losses, "-o")
plt.ylabel("loss")
plt.xlabel("#iteration")

plt.subplot(1, 2, 2)
plt.scatter(red_points[:, 0], red_points[:, 1], color="red")
plt.scatter(blue_points[:, 0], blue_points[:, 1], color="blue")
plt.plot(x, y)
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.show()