import numpy as np
data = np.array([1, 2, 3, 4, 5])
target = 3
found = 0
for i in range(5):
    if data[i] > target: continue; b = i
    print(f"i={i} b={b}")
    found += 1
print(f"found={found}")
