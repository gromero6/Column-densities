import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import os, sys, glob, time, csv

NNorRN = str(sys.argv[1]) # NN (column vs column) or RN (radius vs column)
seed   = int(sys.argv[2])
m      = int(sys.argv[3])
d      = int(sys.argv[4])
graphn = int(sys.argv[5])

NNdirec = "losNvspathN"

if NNorRN == "NN":
    graphsfolder = os.path.join("graphs", NNdirec)
    graphname    = f"losNvspathN_{seed}_{m}_{d}_{graphn}.png"
elif NNorRN == "RN":
    graphsfolder = os.path.join("graphs")
    graphname    = f"ColumnDensity_vs_RadialDistance_{seed}_{m}_{d}_{graphn}.png"
else:
    raise FileNotFoundError(f"No directory exists for the case {NNorRN}, try with NN or NR")

full_graph_path = os.path.join(graphsfolder, graphname)

img = mpimg.imread(full_graph_path)

plt.imshow(img)
plt.show()