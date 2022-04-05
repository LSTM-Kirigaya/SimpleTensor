import os
import SimpleTensor as st

print("default")
os.system("python -u test/train_linear_stable.py -n 10")

for k in st.runtime.init_func:
    print(k)
    os.system("python -u test/train_linear_stable.py -n 10 --init {}".format(k))