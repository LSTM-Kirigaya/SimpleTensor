import numpy as np
from SimpleTensor.constant import EPSILON

def normal_init(shape, mean=1, std=0.01) -> np.ndarray:
    """ submit N(1, 0.01) by default
    """
    return np.random.normal(mean, std, shape)

def uniform_init(shape, input_size=None, output_size=None) -> np.ndarray:
    """ submit to U(-1/sqrt(input_size), 1/sqrt(output_size))
    """
    if input_size is None:
        input_size = shape[0]
    if output_size is None:
        output_size = shape[1]
    return np.random.uniform(
        low=-1 / (np.sqrt(input_size) + EPSILON),
        high=1 / (np.sqrt(output_size) + EPSILON),
        size=shape
    )

def xavier_normal_init(shape, input_size=None, output_size=None) -> np.ndarray:
    # from paper: Understanding the difficulty of training deep feedforward neural networks
    # I recommend this method when you use tanh, sigmoid
    if input_size is None:
        input_size = shape[0]
    if output_size is None:
        output_size = shape[1]
    return np.random.normal(
        loc=0,
        scale=np.sqrt(2 / (input_size + output_size) + EPSILON),
        size=shape
    )

def xavier_uniform_init(shape, input_size=None, output_size=None) -> np.ndarray:
    # from paper: Understanding the difficulty of training deep feedforward neural networks
    # I recommend this method when you use tanh, sigmoid
    if input_size is None:
        input_size = shape[0]
    if output_size is None:
        output_size = shape[1]
    return np.random.uniform(
        low=-np.sqrt(6 / (input_size + output_size) + EPSILON),
        high=np.sqrt(6 / (input_size + output_size) + EPSILON)
    )

def he_normal_init(shape, input_size=None, output_size=None) -> np.ndarray:
    # from paper: Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification
    # I recommend this method when you use ReLU, Leaky ReLU
    if input_size is None:
        input_size = shape[0]
    if output_size is None:
        output_size = shape[1]
    return 

def he_normal_init(shape, input_size=None, output_size=None) -> np.ndarray:
    # from paper: Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification
    # I recommend this method when you use ReLU, Leaky ReLU
    if input_size is None:
        input_size = shape[0]
    if output_size is None:
        output_size = shape[1]



if __name__ == "__main__":
    init = normal_init((10, 10))
    print(init)