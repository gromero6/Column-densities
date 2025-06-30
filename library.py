import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from scipy.spatial import distance

""" Toggle Parameters """

""" Constants and convertion factor """

hydrogen_mass = 1.6735e-24 # gr

# Unit Conversions
km_to_parsec = 1 / 3.085677581e13  # 1 pc in km
pc_to_cm = 3.086 * 10e+18  # cm per parsec
AU_to_cm = 1.496 * 10e+13  # cm per AU
parsec_to_cm3 = 3.086e+18  # cm per parsec

# Physical Constants
mass_unit = 1.99e33  # g
length_unit = 3.086e18  # cm
velocity_unit = 1e5  # cm/s
time_unit = length_unit / velocity_unit  # s
seconds_in_myr = 1e6 * 365.25 * 24 * 3600  # seconds in a million years (Myr)
boltzmann_constant_cgs = 1.380649e-16  # erg/K
grav_constant_cgs = 6.67430e-8  # cm^3/g/s^2
hydrogen_mass = 1.6735e-24  # g

# Code Units Conversion Factors
myrs_to_code_units = seconds_in_myr / time_unit
code_units_to_gr_cm3 = 6.771194847794873e-23  # conversion from code units to g/cm^3
gauss_code_to_gauss_cgs = (1.99e+33/(3.086e+18*100_000.0))**(-1/2)

# ISM Specific Constants
mean_molecular_weight_ism = 2.35  # mean molecular weight of the ISM
gr_cm3_to_nuclei_cm3 = 6.02214076e+23 / (2.35) * 6.771194847794873e-23  # Wilms, 2000 ; Padovani, 2018 ism mean molecular weight is # conversion from g/cm^3 to nuclei/cm^3


""" Arepo Process Methods (written by A. Mayer at MPA July 2024)

(Original Functions Made By A. Mayer (Max Planck Institute) + contributions B. E. Velazquez (University of Texas))
"""
def get_magnetic_field_at_points(x, Bfield, rel_pos):
	n = len(rel_pos[:,0])
	local_fields = np.zeros((n,3))
	for  i in range(n):
		local_fields[i,:] = Bfield[i,:]
	return local_fields

def get_density_at_points(x, Density, Density_grad, rel_pos):
	n = len(rel_pos[:,0])	
	local_densities = np.zeros(n)
	for  i in range(n):
		local_densities[i] = Density[i] + np.dot(Density_grad[i,:], rel_pos[i,:])
	return local_densities

def find_points_and_relative_positions(x, Pos, VoronoiPos):
    dist, cells = spatial.KDTree(Pos[:]).query(x, k=1, workers=-1)
    rel_pos = VoronoiPos[cells] - x
    return dist, cells, rel_pos

def find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos):
	dist, cells, rel_pos = find_points_and_relative_positions(x, Pos, VoronoiPos)
	local_fields = get_magnetic_field_at_points(x, Bfield[cells], rel_pos)
	local_densities = get_density_at_points(x, Density[cells], Density_grad[cells], rel_pos)
	abs_local_fields = np.sqrt(np.sum(local_fields**2,axis=1))
	return local_fields, abs_local_fields, local_densities, cells
	
def Heun_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume, bdirection = None):
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1,(3,1)).T
    CellVol = Volume[cells]
    dx *= ((3/4)*Volume[cells]/np.pi)**(1/3)  
    x_tilde = x + dx[:, np.newaxis] * local_fields_1
    local_fields_2, abs_local_fields_2, local_densities, cells = find_points_and_get_fields(x_tilde, Bfield, Density, Density_grad, Pos, VoronoiPos)
    local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2,(3,1)).T	
    abs_sum_local_fields = np.sqrt(np.sum((local_fields_1 + local_fields_2)**2,axis=1))

    unito = 2*(local_fields_1 + local_fields_2)/abs_sum_local_fields[:, np.newaxis]
    x_final = x + 0.5 * dx[:, np.newaxis] * unito

    bdirection = 0.5*(local_fields_1 + local_fields_2)
    
    return x_final, abs_local_fields_1, local_densities, CellVol

def list_files(directory, ext):
    import os
    # List all files in the specified directory
    all_files = os.listdir(directory)
    # Filter out only .npy files
    files = [directory + f for f in all_files if f.endswith(f'{ext}')]
    return files


""" Process Lines from File into Lists """

def process_line(line):
    """
    Process a line of data containing information about a trajectory.

    Args:
        line (str): Input line containing comma-separated values.

    Returns:
        dict or None: A dictionary containing processed data if the line is valid, otherwise None.
    """
    parts = line.split(',')
    if len(parts) > 1:
        iteration = int(parts[0])
        traj_distance = float(parts[1])
        initial_position = [float(parts[2:5][0]), float(parts[2:5][1]), float(parts[2:5][2])]
        field_magnitude = float(parts[5])
        field_vector = [float(parts[6:9][0]), float(parts[6:9][1]), float(parts[6:9][2])]
        posit_index = [float(parts[9:][0]), float(parts[9:][1]), float(parts[9:][2])]

        data_dict = {
            'iteration': int(iteration),
            'trajectory (s)': traj_distance,
            'Initial Position (r0)': initial_position,
            'field magnitude': field_magnitude,
            'field vector': field_vector,
            'indexes': posit_index
        }

        return data_dict
    else:
        return None
    
""" Energies """

def fibonacci_sphere(samples=100):
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
    y = np.linspace(1 - 1/samples, -1 + 1/samples, samples)  # Even spacing in y
    radius = np.sqrt(1 - y**2)  # Compute radius for each point
    theta = phi * np.arange(samples)  # Angle increment

    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    return np.vstack((x, y, z)).T  # Stack into a (N, 3) array

def pocket_finder(bfield, cycle=0, plot=False):
    """  
    Finds peaks in a given magnetic field array.

    Args:
        bfield (array-like): Array or list of magnetic field magnitudes.
        cycle (int, optional): Cycle number for saving the plot. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple: Contains two tuples:
            - (indexes, peaks): Lists of peak indexes and corresponding peak values.
            - (index_global_max, upline): Indices and value of the global maximum.
    """
    bfield = np.array(bfield)  # Ensure input is a numpy array

    baseline = np.min(bfield)
    upline = np.max(bfield)
    index_global_max = np.where(bfield == upline)[0]
    try:
        idx = index_global_max[0]
    except:
        idx = index_global_max
    upline == bfield[idx]
    ijk = np.argmax(bfield)
    bfield[ijk] = bfield[ijk]*1.001 # if global_max is found in flat region, choose one and scale it 0.001


    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield)):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks +  list(reversed(rpeaks))[1:]
    indexes = lindex + list(reversed(rindex))[1:]
    
    if plot:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        axs.plot(bfield)
        axs.plot(indexes, peaks, "x", color="green")
        axs.plot(indexes, peaks, ":", color="green")
        
        #for idx in index_global_max:
        axs.plot(idx, upline, "x", color="black")
        axs.plot(np.ones_like(bfield) * baseline, "--", color="gray")
        axs.set_xlabel("Index")
        axs.set_ylabel("Field")
        axs.set_title("Actual Field Shape")
        axs.legend(["bfield", "all peaks", "index_global_max", "baseline"])
        axs.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the figure
        plt.savefig(f"./field_shape{cycle}.png")
        plt.close(fig)

    return (indexes, peaks), (index_global_max, upline)

def smooth_pocket_finder(bfield, cycle=0, plot=False):
    """  
    Finds peaks in a given magnetic field array.

    Args:
        bfield (array-like): Array or list of magnetic field magnitudes.
        cycle (int, optional): Cycle number for saving the plot. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple: Contains two tuples:
            - (indexes, peaks): Lists of peak indexes and corresponding peak values.
            - (index_global_max, upline): Indices and value of the global maximum.
    """
    bfield = np.array(bfield)  # Ensure input is a numpy array

    baseline = np.min(bfield)
    upline = np.max(bfield)
    index_global_max = np.where(bfield == upline)[0]
    upline == bfield[index_global_max]
    ijk = np.argmax(bfield)
    bfield[ijk] = bfield[ijk]*1.001 # if global_max is found in flat region, choose one and scale it 0.001

    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield)):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks +  list(reversed(rpeaks))[1:]
    indexes = lindex + list(reversed(rindex))[1:]
    
    if plot:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        axs.plot(bfield)
        axs.plot(indexes, peaks, "x", color="green")
        axs.plot(indexes, peaks, ":", color="green")
        
        axs.plot(np.ones_like(bfield) * baseline, "--", color="gray")
        axs.set_xlabel("Index")
        axs.set_ylabel("Field")
        axs.set_title("Actual Field Shape")
        axs.legend(["bfield", "all peaks", "index_global_max", "baseline"])
        axs.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the figure
        plt.savefig(f"./field_shape{cycle}.png")
        plt.close(fig)

    return (indexes, peaks), (index_global_max, upline)

def find_insertion_point(array, val):

    for i in range(len(array)):
        if val < array[i]:
            return i  # Insert before index i
    return len(array)  # Insert at the end if p_r is greater than or equal to all elements

def find_vector_in_array(radius_vector, x_init):
    """
    Finds the indices of the vector x_init in the multidimensional numpy array radius_vector.
    
    Parameters:
    radius_vector (numpy.ndarray): A multidimensional array with vectors at its entries.
    x_init (numpy.ndarray): The vector to find within radius_vector.
    
    Returns:
    list: A list of tuples, each containing the indices where x_init is found in radius_vector.
    """
    x_init = np.array(x_init)
    return np.argwhere(np.all(radius_vector == x_init, axis=-1))

def magnitude(v1, v2=None):
    if v2 is None:
        return np.linalg.norm(v1)  # Magnitude of a single vector
    else:
        return np.linalg.norm(v1 - v2)  # Distance between two vectors