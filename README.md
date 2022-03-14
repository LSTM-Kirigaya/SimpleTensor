# SimpleTensor
a simple demo to implement a deep learning frame based on static graph

Frame is implemented in `.\SimpleTensor\core.py`. You can even copy all the code directly given to its short code length.

It's `2021.2.18`, I haven't finished `Operation` of CNN, RNN and transformer, which might make some audience disappointed. Fine, give me a chance and I will implement the lovely API after I finish my second track of my album :D

It's `2021.2.18`. Wow, one year has passed and I almost do nothing on the project! Fine, I will reconstruct it with more advanced tech!

Have fun!

---

# TODO

- [ ] support 2D conv
- [ ] support RNN
- [ ] resolve the unstable calcualtion problem
- [ ] reconstruct by ABC
- [ ] reconstruct by C++/PyBind11
- [ ] support JIT
- [ ] lazy module (predefined training pipeline)
- [ ] support ONNX

---

I am going to presenting how the cake is baked instead of being going to tell you how to use it.

```python
from SimpleTensor.core import Placeholder, Variable
from SimpleTensor.core import Session
from SimpleTensor.core import mean_square_error, SGD, Linear

from sklearn.datasets import load_boston 
import matplotlib.pyplot as plt
import numpy as np

X, Y = load_boston(return_X_y=True)

sample_num = X.shape[0]
ratio = 0.8
offline = int(ratio * sample_num)
indexes = np.arange(sample_num)
np.random.shuffle(indexes)

train_X, train_Y = X[indexes[:offline]], Y[indexes[:offline]].reshape([-1, 1])
test_X, test_Y = X[indexes[offline:]], Y[indexes[offline:]].reshape([-1, 1])

X = Placeholder()
Y = Placeholder()

out1 = Linear(13, 8)(X)
out2 = Linear(8, 1)(out1)

loss = mean_square_error(predict=out2, label=Y)

session = Session()
optimizer = SGD(learning_rate=1e-7)

losses = []

for epoch in range(20):
    session.run(root_op=loss, feed_dict={X : train_X, Y : train_Y})
    losses.append(loss.numpy)
    optimizer.minimize(loss)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss:{loss.numpy}")

session.run(root_op=loss, feed_dict={X : test_X, Y : test_Y})
predict = out2.numpy

plt.style.use("seaborn")
plt.subplot(1, 2, 1)
plt.plot(losses, "-o")
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(test_Y.reshape(-1), "r", label="ground truth", alpha=0.5)
plt.plot(predict.reshape(-1), "b", label="predict", alpha=0.5)
plt.legend()
plt.grid(True)


plt.show()

from SimpleTensor.core import _default_graph
for item in _default_graph:
    print(item.__class__)
```

out:
```
[Epoch:0] loss:20611671.255910967
[Epoch:1] loss:844246.2268810662
[Epoch:2] loss:644749.8954289246
[Epoch:3] loss:537370.8130204268
[Epoch:4] loss:455792.21622266097
[Epoch:5] loss:388983.4214624555
[Epoch:6] loss:333234.6495383637
[Epoch:7] loss:286335.4374487398
[Epoch:8] loss:246661.05681635716
[Epoch:9] loss:212946.3762938652
[Epoch:10] loss:184185.53356903407
[Epoch:11] loss:159568.95894414085
[Epoch:12] loss:138438.7688158328
[Epoch:13] loss:120255.86630502032
[Epoch:14] loss:104575.20527502825
[Epoch:15] loss:91026.93924800736
[Epoch:16] loss:79301.88836930264
[Epoch:17] loss:69140.21675196146
[Epoch:18] loss:60322.524301038015
[Epoch:19] loss:52662.773815673216
<class 'SimpleTensor.core.Placeholder'>
<class 'SimpleTensor.core.Placeholder'>
<class 'SimpleTensor.core.Variable'>
<class 'SimpleTensor.core.Variable'>
<class 'SimpleTensor.core.matmul'>
<class 'SimpleTensor.core.add'>
<class 'SimpleTensor.core.Variable'>
<class 'SimpleTensor.core.Variable'>
<class 'SimpleTensor.core.matmul'>
<class 'SimpleTensor.core.add'>
<class 'SimpleTensor.core.minus'>
<class 'SimpleTensor.core.elementwise_pow'>
<class 'SimpleTensor.core.reduce_mean'>
```
![](https://i.loli.net/2021/02/22/YmLKuanDQW2seTp.png)