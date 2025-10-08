import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt
from library import *
import pandas as pd
import numpy.ma as ma

FloatType = np.float64
IntType = np.int32

seed  = int(sys.argv[1])
m     = int(sys.argv[2])
d     = int(sys.argv[3])
cloud = int(sys.argv[4])


datapath  = os.path.join("thesis_los", "ideal", "430")
data_name = f"DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{cloud}.npz"
full_path = os.path.join(datapath, data_name)

data = np.load(full_path)

x_init  = data["x_init_points"]
mean_CD = data["mean_column_densities"]
path_CD = data["pathcolumn"]
full_columns = data["full_columns"]
final_column_density = mean_CD[-1,:]
densities_all = data["densities"]
positions_all = data["positions"]

CD_quotient = np.divide(final_column_density, path_CD)
mask_quotient = np.zeros(m, dtype = bool)
mask_quotient = CD_quotient > 10

filteredmean = final_column_density[mask_quotient]
filteredpath = path_CD[mask_quotient]

print(data.files)
print(final_column_density.size)
print(mask_quotient.size)
print(mask_quotient)

plt.figure(figsize=(10,6))

plt.scatter(filteredpath, filteredmean, s=10, alpha=0.5)

plt.legend()

plt.xscale("log")
plt.yscale("log")


plt.xlabel('B field: $N (cm^{-2})$', fontsize=14)
plt.ylabel('LOS path: $N (cm^{-2})$', fontsize=14)


plt.grid(True, which="both", ls=":")

plt.show()

top5 = np.argsort(CD_quotient)[-5:]
d    = full_columns.shape[2]

