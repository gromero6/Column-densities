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


r_values_los  = []
r_values_path = []
i= 0
while i <16:

    data_directory = os.path.join("thesis_los", case, snap)
    data_name = f"DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{i}.npz"
    full_data_path = os.path.join(data_directory, data_name)
    data = np.load(full_data_path)


    print(data.files)
    output_folder = "./graphs"
    png_name = f"ColumnDensity_vs_RadialDistance_{seed}_{m}_{d}_{i}.png"

    x_init  = data['x_init_points']
    mean_CD = data['mean_column_densities']
    path_CD = data["pathcolumn"]

    final_column_density = mean_CD[-1, :]
    radial_distance_pc = np.linalg.norm(x_init, axis=1)
    radial_distance_cm = radial_distance_pc * pc_to_cm

    plt.figure(figsize=(10, 6))
    #filter LOS data to keep negatives out (all N >0 ut a good practice to have :) )
    los_x  = radial_distance_cm[final_column_density > 0]
    los_y  = final_column_density[final_column_density > 0]
    #now for the B field path
    path_x = radial_distance_cm[path_CD > 0]
    path_y = path_CD[path_CD > 0]
    
    los_slope, los_intercept, los_r, los_p, los_std_err = linregress(np.log10(los_x), np.log10(los_y))
    los_fitline = los_slope * np.log10(los_x) + los_intercept

    path_slope, path_intercept, path_r, path_p, path_std_err = linregress(np.log10(path_x), np.log10(path_y))
    path_fitline = path_slope * np.log10(path_x) + path_intercept

    r_values_los.append(los_r)
    r_values_path.append(path_r)
    
    
    plt.scatter(radial_distance_cm, final_column_density, s=10, alpha=0.5, color = "blue", label = " mean final column density along LOS")
    plt.scatter(radial_distance_cm, path_CD, s = 10, alpha =0.3, color= "red", label = "column density along a B field path")

    plt.plot(los_x, 10**los_fitline, color="black", ls = ":", lw = 1, label=f"LOS Fit: $N \\propto r$ ($R^2={los_r**2:.2f}$)")
    plt.plot(path_x, 10**path_fitline, color = "black", ls = "--", lw = 1, label=f"Path Fit: $N \\propto r$ ($R^2={path_r**2:.2f}$)")

    plt.legend()

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('$cm$', fontsize=14)
    plt.ylabel('$N (cm^{-2})$', fontsize=14)
    plt.title('Column Density vs. Distance', fontsize=16)

    plt.grid(True, which="both", ls=":")
    plt.ylim( (10e20, 10e27) )

    full_path = os.path.join(output_folder, png_name)
    plt.savefig(full_path, dpi=300)
    print("plot saved to: ", full_path)


    i += 1

total_rs = np.column_stack((r_values_los, r_values_path))

print("\n\n--- R-values Table ---")
print("Cloud\tLOS R-value\tPath R-value")
print("---------------------------------------")
for i, (los_r, path_r) in enumerate(total_rs):
    print(f"{i}\t{los_r:.4f}\t\t{path_r:.4f}")
print("---------------------------------------")
