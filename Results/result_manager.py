import numpy as np
import sys

npz = np.load(sys.argv[1])

print(len(npz["t_p_acc"]))
print(npz["t_p_acc"].max())
print(npz["v_p_acc"].max())
print(npz["t_v_loss"].min())
print(npz["v_v_loss"].min())
