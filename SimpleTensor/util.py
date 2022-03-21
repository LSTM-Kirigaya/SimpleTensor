# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.13

from collections.abc import MutableMapping
from typing import Iterator, TypeVar
from colorama import Back, Fore, Style
import numpy as np


KT = TypeVar("KT")   # key   type
VT = TypeVar("VT")   # value type

class Register(MutableMapping):
    __slots__ = "__data"
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self.__data = dict(*args, **kwargs)
    
    def register(self, target):
        def add_register_items(__k, __v):
            if not callable(__v):
                raise ValueError("register object must be callable, but receive {}".format(type(__v)))
            if __k in self.__data:
                back_print(
                    "warning: {} has been registered before, so we will overriden it".format(__v.__name__),
                    color="red")
            self[__k] = __v
            return __v
        return add_register_items(target.__name__, target) if callable(target) else lambda x : add_register_items(target, x)

    def __call__(self, *args, **kwargs):
        return self.register(*args, **kwargs)

    def __setitem__(self, __k : KT, __v : VT) -> None:
        self.__data[__k] = __v
    
    def __getitem__(self, __k : KT) -> VT:
        return self.__data.__getitem__(__k)
    
    def __contains__(self, __k : KT) -> bool:
        return self.__data.__contains__(__k)
    
    def __delitem__(self, __v : VT) -> None:
        return self.__data.__delitem__(__v)
    
    def __len__(self) -> int:
        return self.__data.__len__()
    
    def __iter__(self) -> Iterator[KT]:
        return self.__data.__iter__()

    def __str__(self) -> str:
        return self.__data.__str__()

    def keys(self):
        return self.__data.keys()
    
    def values(self):
        return self.__data.values()
    
    def items(self):
        return self.__data.items()

r = Register()

def back_print(*args, color: str = None) -> None:
    if color is None:
        print(*args)
    else:
        print(getattr(Back, color.upper(), ""), *args, Style.RESET_ALL)

def fore_print(*args, color: str = None) -> None:
    if color is None:
        print(*args)
    else:
        print(getattr(Fore, color.upper(), ""), *args, Style.RESET_ALL)

def numpy_one_hot(y: np.ndarray, class_num: int=None) -> np.ndarray:
    y = y.reshape(-1)
    if class_num is None:
        class_num = len(set(y))
    else:
        y_unique = len(set(y))
        if y_unique < class_num:
            fore_print("#unique value in Y is smaller than class_num, though porgram can still be execuated, I advise you to check Y", color="yellow")
        elif y_unique > class_num:
            raise ValueError("#unique value in Y is greater than class_num!")

    one_hot = np.zeros((y.shape[0], class_num))
    for i in range(y.shape[0]):
        one_hot[i][y[i]] = 1
    return one_hot

