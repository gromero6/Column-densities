import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt
from library import *

start_time = time.time()

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
    dist, cells = spatial.KDTree(Pos[:]).query(x, k=1)
    rel_pos = VoronoiPos[cells] - x
    return dist, cells, rel_pos

def find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos):
	dist, cells, rel_pos = find_points_and_relative_positions(x, Pos, VoronoiPos)
	local_fields = get_magnetic_field_at_points(x, Bfield[cells], rel_pos)
	local_densities = get_density_at_points(x, Density[cells], Density_grad[cells], rel_pos)
	abs_local_fields = np.sqrt(np.sum(local_fields**2,axis=1))
	return local_fields, abs_local_fields, local_densities, cells
	
def Heun_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume):
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1,(3,1)).T
    CellVol = Volume[cells]
    dx *= 0.4*((3/4)*CellVol/np.pi)**(1/3)  
    x_tilde = x + dx[:, np.newaxis] * local_fields_1
    local_fields_2, abs_local_fields_2, local_densities, cells = find_points_and_get_fields(x_tilde, Bfield, Density, Density_grad, Pos, VoronoiPos)
    local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2,(3,1)).T	
    abs_sum_local_fields = np.sqrt(np.sum((local_fields_1 + local_fields_2)**2,axis=1))

    unito = 2*(local_fields_1 + local_fields_2)/abs_sum_local_fields[:, np.newaxis]
    x_final = x + 0.5 * dx[:, np.newaxis] * unito
    kinetic_energy = 0.5*Mass[cells]*np.linalg.norm(Velocities[cells], axis=1)**2
    pressure = Pressure[cells]
    
    return x_final, abs_local_fields_1, local_densities, CellVol, kinetic_energy, pressure

FloatType = np.float64
IntType = np.int32

""" 
python3 los_stats.py 2000 ideal 430 50 S seed > ELOS430TST.txt 2> ELOS430TST_error.txt &
python3 los_stats.py 2000 ideal 430 50 N seed > NLOS430TST.txt 2> NLOS430TST_error.txt &

S : Stability
N : Column densities

"""
if len(sys.argv)>6:
    N                 = int(sys.argv[1])
    case              = str(sys.argv[2]) #ideal/amb
    num_file          = str(sys.argv[3]) 
    max_cycles        = int(sys.argv[4]) 
    try:
        seed              = int(sys.argv[5])
    except:
        seed            = 12345
else:
    N               = 2_000
    case            = 'ideal'
    num_file        = '430'
    max_cycles      = 100
    seed            = 12345

rloc = 10

if case == 'ideal':
    subdirectory = 'ideal_mhd'
elif case == 'amb':
    subdirectory = 'ambipolar_diffusion'
else:
    subdirectory= ''

file_list = glob.glob(f'*.hdf5')
print(file_list)
filename = None

for f in file_list:
    if num_file in f:
        filename = f
if filename == None:
    raise FileNotFoundError

file_path = f'./{case}_cloud_trajectory.txt'

snap = []
time_value = []

with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        snap.append(int(row[0]))  # First column is snap
        time_value.append(float(row[1]))  # Second column is time_value
        if num_file == str(row[0]):
            Center = np.array([float(row[2]),float(row[3]),float(row[4])])

snap_array = np.array(snap)
time_value_array = np.array(time_value)

file_list = glob.glob(f'snap_430.hdf5')
filename = None

for f in file_list:
    if num_file in f:
        filename = f
if filename == None:
    raise FileNotFoundError
snap = filename.split(".")[0][-3:]

new_folder = os.path.join(f"thesis_los/{case}" , snap)
os.makedirs(new_folder, exist_ok=True)

data = h5py.File(filename, 'r')
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Pressure = np.asarray(data['PartType0']['Pressure'], dtype=FloatType)
Velocities = np.asarray(data['PartType0']['Velocities'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))
Volume   = Mass/Density

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

snap = []
time_value = []

with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        if num_file == str(row[0]):
            Center = np.array([float(row[2]),float(row[3]),float(row[4])])
            snap =str(row[0])
            time_value = float(row[1])
            peak_den =  float(row[5])

CloudCord = Center.copy()

print("Center before Centering", Center)

VoronoiPos-=CloudCord
Pos-=CloudCord

for dim in range(3):  # Loop over x, y, z
    pos_from_center = Pos[:, dim]
    boundary_mask = pos_from_center > Boxsize / 2
    Pos[boundary_mask, dim] -= Boxsize
    VoronoiPos[boundary_mask, dim] -= Boxsize

def generate_vectors_in_core(max_cycles, densthresh, rloc=1.0, seed=12345):
    import numpy as np
    from scipy.spatial import cKDTree
    np.random.seed(seed)
    valid_vectors = []
    tree = cKDTree(Pos)
    while len(valid_vectors) < max_cycles:
        points = np.random.uniform(low=-rloc, high=rloc, size=(max_cycles, 3))
        distances = np.linalg.norm(points, axis=1)
        inside_sphere = points[distances <= rloc]
        _, nearest_indices = tree.query(inside_sphere)
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > densthresh
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
    valid_vectors = np.array(valid_vectors)
    random_indices = np.random.choice(len(valid_vectors), max_cycles, replace=False)
    return valid_vectors[random_indices]


"""
x_init =  'm' number of 3D vectors
directions =  '20' unit vectors evenly distributed in a unit sphere

x_init = [
        [a,b,c],
        ]

directions = [
        [d,e,f],
        [d,e,f],
        [d,e,f],
        ]

x_new = x_init + dx*directions

size of x_ init has to increade from its original size times the size of directions
x_init = [
        [a1,b1,c1],
        [a1,b1,c1],
        [a1,b1,c1],
        [a2,b2,c2],
        [a2,b2,c2],
        [a2,b2,c2],
        [a3,b3,c3], 
        [a3,b3,c3],
        [a3,b3,c3],
        ]

size of directions has to increade from its original size times the size of x_init        
directions = [
        [d,e,f],
        [d,e,f],
        [d,e,f],
        ...
        [d,e,f],
        [d,e,f],
        [d,e,f],
        ...
        [d,e,f],
        [d,e,f],
        [d,e,f],
        ]

x_new = x_init + dx*directions

"""

def get_line_of_sight(x_init=None, directions=fibonacci_sphere()):
    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    directions = directions/np.linalg.norm(directions, axis=1)[:, np.newaxis]
    dx = 0.5

    #expand arrays

    x_init = np.repeat(x_init, d, axis=0)
    directions = np.tile(directions, (m, 1))  # Repeat directions for each point in x_init

    total_lines = m*d
    line      = np.zeros((N+1, total_lines,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1, total_lines))
    densities = np.zeros((N+1, total_lines))
    volumes   = np.zeros((N+1, total_lines))
    threshold = np.zeros((total_lines,)).astype(int) # one value for each


    line_rev=np.zeros((N+1,total_lines,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,total_lines))
    densities_rev = np.zeros((N+1,total_lines))
    volumes_rev   = np.zeros((N+1,total_lines))
    threshold_rev = np.zeros((total_lines,)).astype(int) # one value for each

    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init 

    x = x_init.copy()

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x_init, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]
    densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] * gr_cm3_to_nuclei_cm3
    k=0

    mask  = dens > 100# 1 if not finished
    un_masked = np.logical_not(mask) # 1 if finished

    while np.any(mask):

        mask = dens > 100  # True if continue
        un_masked = np.logical_not(mask)

        #aux = x[un_masked]

        # Perform Heun step and update values
        _, bfield, dens, vol, ke, pressure = Heun_step(
            x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
        )
        
        mass_dens = dens * code_units_to_gr_cm3
        pressure *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens *= gr_cm3_to_nuclei_cm3
        
        #vol[un_masked] = 0
        print( np.log10(dens[:1]))
        
        non_zero = vol > 0
        if len(vol[non_zero]) == 0:
            break

        dx_vec = np.min(((4 / 3) * vol[non_zero] / np.pi) ** (1 / 3))  # Increment step size

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100

        x += dx_vec * directions
        
        #x[un_masked] = aux # all lines that have reached threshold are not to be updated

        line[k+1,:,:]    = x
        densities[k+1,:] = dens
        bfields[k+1,:] = bfield

        if np.all(un_masked):
            print("All values are False: means all density < 10^2")
            break

        k += 1
    
    threshold = threshold.astype(int)
    
    x = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x_init, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]

    densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    
    k=0
    
    non_zero_rev = vol > 0

    mask_rev = dens > 100
    un_masked_rev = np.logical_not(mask_rev)

    while np.any((mask_rev)):

        mask_rev = dens > 100  # True if continue
        un_masked_rev = np.logical_not(mask_rev)

        # Perform Heun step and update values
        _, bfield, dens, vol, ke, pressure = Heun_step(
            x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
        )
        
        pressure *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens *= gr_cm3_to_nuclei_cm3
        
        #vol[un_masked_rev] = 0
        print(x[0], np.log10(dens[0]))

        non_zero_rev = vol > 0
        if len(vol[non_zero_rev]) == 0:
            break

        dx_vec = np.min(((4 / 3) * vol[non_zero] / np.pi) ** (1 / 3))  # Increment step size

        threshold_rev += mask.astype(int)  # Increment threshold count only for values still above 100

        x -= dx_vec * directions

        line_rev[k+1,:,:]    = x
        densities_rev[k+1,:] = dens
        bfields_rev[k+1,:] = bfield

        if np.all(un_masked_rev):
            print("All values are False: means all density < 10^2")
            break

        k += 1

    # updated_mask = np.logical_not(np.logical_and(mask, mask_rev))
    
    #threshold = threshold[updated_mask].astype(int)
    threshold = threshold.astype(int)
    #threshold_rev = threshold_rev[updated_mask].astype(int)
    threshold_rev = threshold_rev.astype(int)

    # Apply updated_mask to the second axis of (N+1, m, 3) or (N+1, m) arrays
    # line = line[:, updated_mask, :]  # Mask applied to the second dimension (m)
    # densities = densities[:, updated_mask]  # Mask applied to second dimension (m)

    # Apply to the reverse arrays in the same way
    # line_rev = line_rev[:, updated_mask, :]
    # densities_rev = densities_rev[:, updated_mask]

    radius_vector = np.append(line_rev[::-1, :, :], line[1:,:,:], axis=0)
    numb_densities = np.append(densities_rev[::-1, :], densities[1:,:], axis=0)
    magnetic_field = np.append(bfields_rev[::-1, :], bfields[1:,:], axis=0)

    trajectory = np.zeros_like(numb_densities)
    column = np.zeros_like(numb_densities)

    print("Surviving lines: ", m, "out of: ", max_cycles)

    for _n in range(radius_vector.shape[1]):  # Iterate over the first dimension
        print("Line: ", _n, " Size: ", radius_vector[:, _n, 0].shape)
        prev = radius_vector[0, _n, :]
        trajectory[0, _n] = 0  # Initialize first row
        column[0, _n] = 0      # Initialize first row
        
        for k in range(1, radius_vector.shape[0]):  # Start from k = 1 to avoid indexing errors            
            cur = radius_vector[k, _n, :]
            diff_rj_ri = magnitude(cur - prev)  # Vector subtraction before calculating magnitude

            trajectory[k, _n] = trajectory[k-1, _n] + diff_rj_ri            
            column[k, _n] = column[k-1, _n] + numb_densities[k, _n] * diff_rj_ri            
            
            prev = cur  # Store current point as previous point

    trajectory      *= 1.0#pc_to_cm #* 3.086e+18                                # from Parsec to cm

    return radius_vector, trajectory, numb_densities, [threshold, threshold_rev], column


print("Steps in Simulation: ", N)
print("Boxsize            : ", Boxsize)
print("Smallest Volume    : ", Volume[np.argmin(Volume)])
print("Biggest  Volume    : ", Volume[np.argmax(Volume)])
print(f"Smallest Density  : {Density[np.argmin(Density)]}")
print(f"Biggest  Density  : {Density[np.argmax(Density)]}")

print("Elapsed Time: ", (time.time() - start_time)/60.)

os.makedirs(new_folder, exist_ok=True)

densthresh = 100



if __name__=='__main__':
    x_init = generate_vectors_in_core(max_cycles, densthresh, rloc, seed)
    directions = fibonacci_sphere()
    m = x_init.shape[0] # number of target points
    d = directions.shape[0] # number of directions
    total_lines = m*d
    print('Directions provided by B field at point')
    radius_vector, trajectory, numb_densities, th, column = get_line_of_sight(x_init, directions)
    threshold, threshold_rev = th
    np.savez(os.path.join(new_folder, f"DataBundle{seed}.npz"),
            thresholds=threshold,
            thresholds_rev=threshold_rev,
            column_densities=column,
            positions=radius_vector,
            number_densities=numb_densities)

print(total_lines, "lines of sight generated for all points")

