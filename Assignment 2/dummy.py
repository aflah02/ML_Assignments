import numpy as np

for i in range(8):
    for j in range(8):
        if i & j == 0:
            print(i, j)