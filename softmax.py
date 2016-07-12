import numpy as np
import math as math


def softmax(scores):
  return np.exp(scores) / np.sum(np.exp(scores),axis=0) 


scores=[1.0,2.0,3.0]
print(softmax(scores))

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

print(softmax(scores))








