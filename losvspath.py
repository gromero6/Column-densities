import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt
from scipy.stats import linregress
from library import *

seed  = input("Enter seed: ")
m     = input("Enter number of points: ")
d     = input("Enter Number of lines of sight per point: ")
case  = input("Enter case (ideal/amb): ")
snap  = input("Enter snapshot number: ")


coordirec = os.path.join("clouds")
cornames  = f"{case}_clouds.txt"
full_cord_path = os.path.join(coordirec, cornames)
df = pd.read_csv(full_cord_path)

i = 0
while i <20:

    cloud_data   = df.iloc[i]
    cloud_number = cloud_data["index"]
    peak_density = cloud_data["Peak_Density"]
    x = cloud_data["CloudCord_X"]
    y = cloud_data["CloudCord_Y"]
    z = cloud_data["CloudCord_Z"]

    data_coordinates = np.load
    data_directory = os.path.join("thesis_los", case, snap)
    data_name = f"DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{i}.npz"
    full_data_path = os.path.join(data_directory, data_name)
    data = np.load(full_data_path)


    print(data.files)
    output_folder = "./graphs/losNvspathN"
    png_name = f"losNvspathN_{seed}_{m}_{d}_{i}.png"

    mean_CD = data['mean_column_densities']
    path_CD = data["pathcolumn"]

    final_column_density = mean_CD[-1, :]

    plt.figure(figsize=(10, 6))
    
    
    plt.scatter(path_CD, final_column_density, s=10, alpha=0.5)

    plt.legend()

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('$B$ field: $N (cm^{-2})$', fontsize=14)
    plt.ylabel('LOS: $N (cm^{-2})$', fontsize=14)
    
    title_text = f"Column Density following the B field vs. Column density following the LOS, Cloud {int(cloud_number)}"
    plt.title(title_text, fontsize=16)

    plt.grid(True, which="both", ls=":")

    info_text = (
    f"Center Coordinates: ({x:.2f}, {y:.2f}, {z:.2f})\n"
    f"Peak Density: {peak_density:.2f}"
    )
    plt.figtext(0.15, 0.02, info_text, fontsize=10, ha='left')

    full_path = os.path.join(output_folder, png_name)
    plt.savefig(full_path, dpi=300)
    print("plot saved to: ", full_path)


    i += 1


