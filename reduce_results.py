#%%
K = 7
from functools import reduce
import json
import numpy as np
with open('results.json', 'r') as file:
  data = json.loads(file.read())

def reducer(x, y):
  xres = np.mean([
    np.mean(x['war_val_aucs'][-K:]),
    np.mean(x['ff_val_aucs'][-K:])
  ])
  yres = np.mean([
    np.mean(y['war_val_aucs'][-K:]),
    np.mean(y['ff_val_aucs'][-K:])
  ])
  return x if xres>yres else y 
val = reduce(
  reducer, data
)
print(val)