import numpy as np
import math as math


def softmax(scores):
  d=np.ndim(scores)

  cols=np.shape(scores)[0]
  rows=0

  if d!=1:
    scores=scores.astype(float)

    rows=np.shape(scores)[1]

    for r in range(rows):
      t=0
      for c in range(cols):
        t=t + math.exp(scores[c][r])

      for c in range(cols):
        v=math.exp(scores[c][r])/t
        scores[c][r]=v


  else:
    t=0
    for ix in range(cols):
       t=t + math.exp(scores[ix])
    for ix in range(cols):
      scores[ix]=math.exp(scores[ix]) /t

  return scores 


scores=[1.0,2.0,3.0]
print(softmax(scores))

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

print(softmax(scores))








