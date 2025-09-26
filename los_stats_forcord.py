import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt
from library import *
from scipy.spatial import KDTree

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

# cache-ing spatial.cKDTree(Pos[:]).query(x, k=1)
_cached_tree = None
_cached_pos = None

def find_points_and_relative_positions(x, Pos, VoronoiPos):
    global _cached_tree, _cached_pos
    if _cached_tree is None or not np.array_equal(Pos, _cached_pos):
        _cached_tree = KDTree(Pos)
        _cached_pos = Pos.copy()
    
    dist, cells = _cached_tree.query(x, k=1, workers=-1)
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
    max_cycles        = int(sys.argv[4]) #numero de puntos
    nd                = int(sys.argv[5]) #numero de direcciónes 
    try:
        seed              = int(sys.argv[6])
    except:
        seed            = 12345
else:
    N               = 2000 #number of steps
    case            = 'ideal' #ideal or ambipolar
    num_file        = '430' #snapshot number
    max_cycles      = 100 # number of x init points generated
    nd              = 10 #directions nunmber of LOS
    seed            = 12345

rloc = 0.1 # radius of the sphere in which the points are generated

#it searches for the directory, ideal or ambipolar
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

#it looks for the path and file where the cloud trajectories are
file_path = os.path.join(".", f"{case}_cloud_trajectory.txt")
clouds_file_path = os.path.join("clouds", f"{case}_clouds.txt")

snap = []
time_value = []


file_list = glob.glob(f'snap_430.hdf5')
filename = None

#troubleshoots, if it doesn't fin the filename, it shows an error
for f in file_list:
    if num_file in f:
        filename = f
if filename == None:
    raise FileNotFoundError
snap = filename.split(".")[0][-3:]

#creates the folder for the saved data
new_folder = os.path.join("thesis_los" , case, snap)
os.makedirs(new_folder, exist_ok=True)

data = h5py.File(filename, 'r')
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype into the arrays that the functions will use
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

#not used in the code
snap = []
time_value = []

#this function is used for finding the cloud center and defining it for each cloud coordinate do that the code can use it in the loop
def clouds_center(clouds_file_path, num_file): #it takes as argument the file number and the path
    centers_list = [] #initializes an empty array that will storage the centers
    found = False #it works with a while loop that will stop when the bool "found" is true
    with open(clouds_file_path, mode='r') as file:
        csv_reader = csv.DictReader(file) #csv reader to read the file and look in it
        for row in csv_reader:
            if int(row["snap"]) == int(num_file):
                centers_list.append([float(row["CloudCord_X"]), float(row["CloudCord_Y"]), float(row["CloudCord_Z"])]) #for each row it takes the cloud coordinate and appends it to the array (number of clouds vs 3 array)
                found = True
    if not found:
        raise ValueError(f"No clouds found for snapshot {num_file}")
    
    return np.array(centers_list) #returns the list of centers

#this function generates vectors from the core to the part where the x init points will be generated, it uses the density threashold so that the x init points are not generated somwhere where the density is lower than the density threashold
def generate_vectors_in_core(max_cycles, densthresh, Pos, rloc=1.0, seed=12345): 
    import numpy as np
    from scipy.spatial import KDTree
    np.random.seed(seed) #it uses the seed such that it generates the same vectors each run unless the seed changes
    valid_vectors = [] 
    tree = KDTree(Pos) #organizes the positions inside the cloud for a kdtree so that it's easy to seach for them 
    while len(valid_vectors) < max_cycles: #it will operate while there's less valid vectors than points should be
        points = np.random.uniform(low=-rloc, high=rloc, size=(max_cycles, 3)) #it generates points in a cube of side 2rloc 
        distances = np.linalg.norm(points, axis=1) #finds the distance between the points and the center
        inside_sphere = points[distances <= rloc] #to keep the points inside the sphere, it filters out the points whose distance is larger than the rloc (radius of the cloud)
        _, nearest_indices = tree.query(inside_sphere) 
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > densthresh #creates a mask for the points in Pos which have a densuty bigger than the densitythreashold
        valid_points = inside_sphere[valid_mask] #filters out for points with density * gr_cm3_to_nuclei_cm3 > densthresh
        valid_vectors.extend(valid_points) #add each of those vectors inside the valid_vectors array
    valid_vectors = np.array(valid_vectors) #turns the valid_vectors array into a numpy array
    random_indices = np.random.choice(len(valid_vectors), max_cycles, replace=False) 
    return valid_vectors[random_indices] #returns the points in a mixed order


def get_line_of_sight(x_init=None, directions=fibonacci_sphere(), Pos=None, VoronoiPos=None):
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

    while np.any(mask) and k < N:

        mask = dens > 100  # True if continue
        un_masked = np.logical_not(mask)

        aux = x[un_masked]

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


        # --- freeze stopped rays ---
        x_old = x.copy()
        x += dx_vec * directions
        x[un_masked] = x_old[un_masked]
        

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

    while np.any((mask_rev)) and k < N:

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

        threshold_rev += mask_rev.astype(int)  # Increment threshold count only for values still above 100


        # --- freeze stopped rays ---
        x_old = x.copy()
        x -= dx_vec * directions
        x[un_masked_rev] = x_old[un_masked_rev]

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
            diff_rj_ri = magnitude(cur - prev) * pc_to_cm # Vector subtraction before calculating magnitude in cm

            trajectory[k, _n] = trajectory[k-1, _n] + diff_rj_ri            
            column[k, _n] = column[k-1, _n] + numb_densities[k, _n] * diff_rj_ri            
            
            prev = cur  # Store current point as previous point

    #trajectory      *= pc_to_cm #* 3.086e+18                                # from Parsec to cm

    return radius_vector, trajectory, numb_densities, [threshold, threshold_rev], column




def get_B_field_column_density(
    x_init,
    Bfield,
    Density,
    densthresh,
    N,
    max_cycles,
    Density_grad,
    Volume,
    VoronoiPos,
    Pos
):
    """
    Calculates the column density over magnetic field lines for all x_init points
    """
    
    BDtotal = np.zeros(max_cycles)
    column_fwd = np.zeros(max_cycles)
    column_bck = np.zeros(max_cycles)
    current_fwd = x_init.copy()
    current_bck = x_init.copy()
    
    mask_fwd = np.ones(max_cycles, dtype=bool)
    mask_bck = np.ones(max_cycles, dtype=bool)

    for i in range(N):
        if np.any(mask_fwd):
            
            # only for active mask poitns
            active_fwd = current_fwd[mask_fwd]
            
            _, _, local_densities_fwd, _ = find_points_and_get_fields(
                active_fwd, Bfield, Density, Density_grad, Pos, VoronoiPos
            )
            
            local_densities_fwd *= gr_cm3_to_nuclei_cm3
            new_mask_fwd = local_densities_fwd > densthresh

            local_densities_fwd[~new_mask_fwd] = 0
            
            next_fwd, _, _, _, _, _ = Heun_step(
                active_fwd, np.ones(len(active_fwd)) * 0.5, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
            )

            distance_traveled_fwd = np.linalg.norm(next_fwd - active_fwd, axis=1) * pc_to_cm

            column_fwd[mask_fwd] += local_densities_fwd * distance_traveled_fwd
            current_fwd[mask_fwd] = next_fwd

            mask_fwd[mask_fwd] = new_mask_fwd
        #backwards
        if np.any(mask_bck):
            
    
            active_bck = current_bck[mask_bck]
            
            _, _, local_densities_bck, _ = find_points_and_get_fields(
                active_bck, Bfield, Density, Density_grad, Pos, VoronoiPos
            )
            
            local_densities_bck *= gr_cm3_to_nuclei_cm3

            # above threashhold
            new_mask_bck = local_densities_bck > densthresh
            
            local_densities_bck[~new_mask_bck] = 0 #those that dont go above the threashold become zero
            
        
            next_bck, _, _, _, _, _ = Heun_step(
                active_bck, np.ones(len(active_bck)) * -0.5, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
            )
            
            distance_traveled_bck = np.linalg.norm(next_bck - active_bck, axis=1) * pc_to_cm
            
            column_bck[mask_bck] += local_densities_bck * distance_traveled_bck
            current_bck[mask_bck] = next_bck
            
            mask_bck[mask_bck] = new_mask_bck
            
        # if both masks false
        if not np.any(mask_fwd) and not np.any(mask_bck):
            break
            
    BDtotal = column_fwd + column_bck
    return BDtotal

print("Simulation Parameters:")
print("Case               : ", case)
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
    cloud_centers = clouds_center(clouds_file_path, num_file)
    num_clouds = cloud_centers.shape[0]

    for i in range(num_clouds):
    
        Pos_copy = Pos.copy()
        VoronoiPos_copy = VoronoiPos.copy()
        
        print("Centering on cloud:", i)
        print("Coordinates:", cloud_centers[i])
        
        # create a copy for each cloud that it doesn't re-center previous coordinates
        Pos_copy -= cloud_centers[i]
        VoronoiPos_copy -= cloud_centers[i]

        for dim in range(3):  # Loop over x, y, z
            pos_from_center = Pos_copy[:, dim]
            boundary_mask = pos_from_center > Boxsize / 2
            Pos_copy[boundary_mask, dim] -= Boxsize
            VoronoiPos_copy[boundary_mask, dim] -= Boxsize
            
            boundary_mask = pos_from_center < -Boxsize / 2
            Pos_copy[boundary_mask, dim] += Boxsize
            VoronoiPos_copy[boundary_mask, dim] += Boxsize
        
        x_init = generate_vectors_in_core(max_cycles, densthresh, Pos_copy, rloc, seed)
        directions = fibonacci_sphere(nd)
        m = x_init.shape[0] # number of target points
        d = directions.shape[0] # number of directions
        total_lines = m*d
        
        print(total_lines, "lines of sight generated for all points")
        print("No. of starting positions:", x_init.shape)
        print("No. of directions:", directions.shape)
        print('Directions provided by the LOS at points')

        radius_vector, trajectory, numb_densities, th, column = get_line_of_sight(x_init, directions, Pos=Pos_copy, VoronoiPos=VoronoiPos_copy)
        threshold, threshold_rev = th

        BDtotal = get_B_field_column_density(x_init,Bfield,Density,densthresh,N,max_cycles,Density_grad, Volume, VoronoiPos = VoronoiPos_copy, Pos = Pos_copy)

        column_reshaped = column.reshape(column.shape[0],m,d) #separates the column densities per point, per directions
        mean_column_per_point = np.mean(column_reshaped, axis= 2) #takes the mean over the directions 
        np.savez(os.path.join(new_folder, f"DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{i}.npz"),
            densities             = numb_densities,
            positions             = radius_vector, 
            mean_column_densities = mean_column_per_point,
            x_init_points         = x_init,
            snapshot_number       = int(num_file),
            pathcolumn            = BDtotal,
            full_columns          = column_reshaped,
            )