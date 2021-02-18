# SimpleTensor
a simple demo to implement a deep learning frame based on static graph

Frame is implemented in `.\SimpleTensor\core.py`. You can even copy all the code directly given to its short code length.

It's `2021.2.18`, I haven't finished `Operation` of CNN, RNN and transformer, which might make some audience disappointed. Fine, give me a chance and I will implement the lovely API after I finish my second track of my album :D

---

I am going to presenting how the cake is baked instead of being going to tell you how to use it.

```python
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
```

out:
```
[Epoch:0] loss:189.08651121931004
[Epoch:1] loss:62.17352146053242
[Epoch:2] loss:77.76117097754383
[Epoch:3] loss:61.119402524423364
[Epoch:4] loss:75.69167935944368
[Epoch:5] loss:63.24291578590535
[Epoch:6] loss:74.20764277941905
[Epoch:7] loss:64.71041241225493
[Epoch:8] loss:73.11469754695592
[Epoch:9] loss:65.77633034532849
[Epoch:6] loss:74.20764277941905
[Epoch:7] loss:64.71041241225493
[Epoch:8] loss:73.11469754695592
[Epoch:9] loss:65.77633034532849
```