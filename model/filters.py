import numpy as np


BLUR_FILTER=np.array([
    [np.ones(shape=(3,3)),np.zeros(shape=(3,3)),np.zeros(shape=(3,3))],
    [np.zeros(shape=(3,3)),np.ones(shape=(3,3)),np.zeros(shape=(3,3))],
    [np.zeros(shape=(3,3)),np.zeros(shape=(3,3)),np.ones(shape=(3,3))]
])/9

BLUR_FILTER=np.array([
    np.ones(shape=(3,3)),np.ones(shape=(3,3)),np.ones(shape=(3,3))
])/9



if __name__=='__main__':
    BLUR_FILTER=np.pad(BLUR_FILTER,(1,))[1:-1]
    print(BLUR_FILTER)

