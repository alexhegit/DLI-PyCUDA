import math
from numba import cuda, vectorize

normalized = cuda.device_array(shape=(n,), dtype=np.float32)
weighted = cuda.device_array(shape=(n,), dtype=np.float32)
#activated = cuda.device_array(shape=(n,), dtype=np.float32)
SOLUTION = cuda.device_array(shape=(n,), dtype=np.float32)

@vectorize(['float32(float32)'], target='cuda')
def normalize(grayscales):
    return grayscales / 255

@vectorize(['float32(float32, float32)'], target='cuda')
def weigh(values, weights):
    return values * weights

@vectorize(['float32(float32)'], target='cuda')
def activate(values):
    return ( math.exp(values) - math.exp(-values) ) / ( math.exp(values) + math.exp(-values) )