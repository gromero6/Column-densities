import numpy as np
import os, time
import matplotlib as plt
from library import *

import pandas as pd

case = input('ideal or amb: ')
snapshot = input('snapshot: ')
m = input('number of points')
d = input('los directions: ')
cloudnum = input('which clouds?')
seed = input('which cloud?')

databundlepath = os.path.join('thesis_los', case, snapshot)
databundlename = f'DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{cloudnum}.npz'
full_path_bundle = os.path.join(databundlepath, databundlename)


data = np.load(full_path_bundle)

cloud_direc = os.path.join('clouds')
coordinates = f'{case}_clouds.txt'
full_cloud_path = os.path.join(cloud_direc, coordinates)
df = pd.read_csv(full_cloud_path)

#core coordinates
cloud_data = df.iloc[full_cloud_path]
peak_density = cloud_data['Peak_Density']
xcore = cloud_data['CloudCord_X']
ycore = cloud_data['CloudCord_Y']
zcore = cloud_data['CloudCord_Z']

print(data)







