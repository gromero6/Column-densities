import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt
from library import *
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm
import numpy.ma as ma

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
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1,(3,1)).T #normalize the local fields
    CellVol = Volume[cells] #volume of the cells where the points are located
    dx *= 0.1*((3/4)*CellVol/np.pi)**(1/3)  #update step size based on cell volume
    x_tilde = x + dx[:, np.newaxis] * local_fields_1 #predictor step
    local_fields_2, abs_local_fields_2, local_densities, cells = find_points_and_get_fields(x_tilde, Bfield, Density, Density_grad, Pos, VoronoiPos) #get fields at predicted position
    local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2,(3,1)).T    #normalize the local fields at predicted position
    abs_sum_local_fields = np.sqrt(np.sum((local_fields_1 + local_fields_2)**2,axis=1)) #magnitude of the sum of the two local fields

    unito = 2*(local_fields_1 + local_fields_2)/abs_sum_local_fields[:, np.newaxis] #unit vector in the direction of the average field
    x_final = x + 0.5 * dx[:, np.newaxis] * unito #corrector step
    kinetic_energy = 0.5*Mass[cells]*np.linalg.norm(Velocities[cells], axis=1)**2 #kinetic energy calculation
    pressure = Pressure[cells] #pressure at the cells
    
    return x_final, abs_local_fields_1, local_densities, CellVol, kinetic_energy, pressure

def Euler_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume, bdirection=None):

    # Get local fields and densities at the current position
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    # Normalize the local fields
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1, (3, 1)).T

    # Update step size based on cell volume
    CellVol = Volume[cells]
    dx *= 0.1*((3 / 4) * Volume[cells] / np.pi) ** (1 / 3)

    # Compute the final position using the Euler method
    x_final = x + dx[:, np.newaxis] * local_fields_1

    # Update the magnetic field direction 
    bdirection = local_fields_1



    return x_final, abs_local_fields_1, local_densities, CellVol


FloatType = np.float64
IntType = np.int32

""" 
python3 los_stats_forcord.py 2000 ideal 430 1000 20 12345

S : Stability
N : Column densities

"""
if len(sys.argv)>6:                                                                                                                                                                                               
    N                 = int(sys.argv[1])
    case              = str(sys.argv[2]) #ideal/amb
    num_file          = str(sys.argv[3]) 
    max_cycles        = int(sys.argv[4]) #numero de puntos
    nd                = int(sys.argv[5]) #numero de direcciónes 
    compute              = int(sys.argv[6])

else:
    N               = 2000 #number of steps
    case            = 'ideal' #ideal or ambipolar
    num_file        = '430' #snapshot number
    max_cycles      = 100 # number of x init points generated
    nd              = 10 #directions nunmber of LOS

rloc = 0.1 # radius of the sphere in which the points are generated
seed = 12345

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


def column_density_LOS_Bfield_clouds():
    full_CD_calculations_time_average = np.zeros((2,num_clouds))
    full_CD_calculations_time         = np.zeros((2,))
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

        start_time_heun = time.perf_counter()
        BD_total_heun = get_B_field_column_density(x_init,Bfield,Density,densthresh,N,max_cycles,Density_grad, Volume, VoronoiPos = VoronoiPos_copy, Pos = Pos_copy)
        end_time_heun = time.perf_counter()

        full_time_heun_CDs = end_time_heun - start_time_heun
        
        start_time_euler = time.perf_counter()
        BD_total_euler = get_B_field_column_density_euler(x_init,Bfield,Density,densthresh,N,max_cycles,Density_grad, Volume, VoronoiPos = VoronoiPos_copy, Pos = Pos_copy)
        end_time_euler = time.perf_counter()

        full_time_euler_CDs = end_time_euler - start_time_euler

        full_CD_calculations_time[0,i] = full_time_heun_CDs
        full_CD_calculations_time[1,i] = full_time_euler_CDs

        column_reshaped = column.reshape(column.shape[0],m,d) #separates the column densities per point, per directions
        mean_column_per_point = np.mean(column_reshaped, axis= 2) #takes the mean over the directions 
        np.savez(os.path.join(new_folder, f"DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{i}.npz"),
            trajectories          = trajectory,
            densities             = numb_densities,
            positions             = radius_vector, 
            mean_column_densities = mean_column_per_point,
            x_init_points         = x_init,
            snapshot_number       = int(num_file),
            pathcolumn_heun       = BD_total_heun,
            pathcolumn_euler      = BD_total_euler,
            full_columns          = column_reshaped,
            CD_times              = full_CD_calculations_time,
            )
        
    average_time_heun = np.mean(full_CD_calculations_time, axis=0)
    average_time_euler = np.mean(full_CD_calculations_time, axis=1) 

    print("Average time for full CD calculations Heun: ", average_time_heun)
    print("Average time for full CD calculations Euler: ", average_time_euler)


def one_cloud_N(i):
    
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

    radius_vector, trajectory, numb_densities, th, column, total_distance_LOS, total_number_density_LOS = get_line_of_sight(x_init, directions, Pos=Pos_copy, VoronoiPos=VoronoiPos_copy)
    threshold, threshold_rev = th

    start_time_heun = time.perf_counter()
    BD_total_heun, posBheun_fwd, ndBheun_fwd, posBheun_bck, ndBheun_bck = get_B_field_column_density(x_init,Bfield,Density,densthresh,N,max_cycles,Density_grad, Volume, VoronoiPos = VoronoiPos_copy, Pos = Pos_copy)
    end_time_heun = time.perf_counter()

    full_time_heun_CDs = end_time_heun - start_time_heun
        
    start_time_euler = time.perf_counter()
    BD_total_euler, posBeuler_fwd, ndBeuler_fwd, posBeuler_bck, ndBeuler_bck = get_B_field_column_density_euler(x_init,Bfield,Density,densthresh,N,max_cycles,Density_grad, Volume, VoronoiPos = VoronoiPos_copy, Pos = Pos_copy)
    end_time_euler = time.perf_counter()

    full_time_euler_CDs = end_time_euler - start_time_euler

    print(posBheun_fwd.shape)
    print(posBeuler_fwd.shape)

    column_reshaped = column.reshape(column.shape[0],m,d) #separates the column densities per point, per directions
    mean_column_per_point = np.mean(column_reshaped, axis= 2) #takes the mean over the directions 
    np.savez(os.path.join(new_folder, f"DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{i}.npz"),
        trajectories          = trajectory,
        densities             = numb_densities,
        positions             = radius_vector, 
        mean_column_densities = mean_column_per_point,
        x_init_points         = x_init,
        snapshot_number       = int(num_file),
        pathcolumn_heun       = BD_total_heun,
        pathcolumn_euler      = BD_total_euler,
        full_columns          = column_reshaped,
        distance_LOS         = total_distance_LOS,
        numberdensity_LOS     = total_number_density_LOS,
        posBheun_fwd         = posBheun_fwd,
        ndBheun_fwd          = ndBheun_fwd,
        posBheun_bck         = posBheun_bck,
        ndBheun_bck          = ndBheun_bck,
        posBeuler_fwd        = posBeuler_fwd,
        ndBeuler_fwd         = ndBeuler_fwd,
        posBeuler_bck        = posBeuler_bck,
        ndBeuler_bck         = ndBeuler_bck,
        )

def get_line_of_sight(x_init=None, directions=fibonacci_sphere(), Pos=None, VoronoiPos=None):
    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    directions = directions/np.linalg.norm(directions, axis=1)[:, np.newaxis]
    dx = 0.5

    #expand arrays
    d = directions.shape[0]
    m = x_init.shape[0] # number of target points
    x_init = np.repeat(x_init, d, axis=0)
    directions = np.tile(directions, (m, 1))  # Repeat directions for each point in x_init


    total_lines = m*d
    line      = np.zeros((N+1, total_lines,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1, total_lines))
    densities = np.zeros((N+1, total_lines))
    volumes   = np.zeros((N+1, total_lines))
    threshold = np.zeros((total_lines,)).astype(int) # one value for each

    # Arrays for distances and number densities (forward and backward)
    distance_fwd = np.zeros((N, m, d))
    number_density_fwd = np.zeros((N, m, d))
    distance_bck = np.zeros((N, m, d))
    number_density_bck = np.zeros((N, m, d))

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

        active_x = x[mask]
    

        # Perform Heun step and update values
        _, bfield, dens_sub, vol, ke, pressure = Heun_step(
            active_x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
        )
        
        still_active_mask = dens_sub>100

        mass_dens = dens * code_units_to_gr_cm3
        pressure *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens_sub *= gr_cm3_to_nuclei_cm3

        # Calculate distances and update arrays
        distance_traveled = np.linalg.norm(active_x - x[mask], axis=1) * pc_to_cm

       
        act_ind = np.where(mask.reshape(m,d))
        # Update distance and number density arrays directly
        distance_fwd[k][act_ind] = distance_traveled
        number_density_fwd[k][act_ind] = dens_sub


        #vol[un_masked] = 0
        print( np.log10(dens[:1]))
        
        non_zero = vol > 0
        if len(vol[non_zero]) == 0:
            break
        
        active_vol = vol[still_active_mask]

        # Only consider active volumes that are non-zero
        volumes_to_minimize = active_vol[active_vol > 0] 

        if len(volumes_to_minimize) == 0:
            # If there are no active, non-zero volumes left, break the loop 
            # as there are no rays to propagate further based on this step size.
            break 
        

        dx_vec = np.min(((4 / 3) * volumes_to_minimize / np.pi) ** (1 / 3))  # Increment step size

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100

        active_indices = np.where(mask)[0]
        still_active_indices = active_indices[still_active_mask]

        new_active_mask = np.zeros_like(mask, dtype=bool)
        new_active_mask[still_active_indices] = True

        # --- freeze stopped rays ---

        x[new_active_mask] += dx_vec * directions[new_active_mask]

        # Step 1: First, carry over the complete state from the previous step (k).
        # This ensures that any line that becomes inactive in this step retains its last active value.
        line[k+1,:,:]      = line[k,:,:]
        densities[k+1,:]   = densities[k,:]
        bfields[k+1,:]     = bfields[k,:]

        # Step 2: Now, use the NEW mask to overwrite the data for ONLY the active lines.
        line[k+1, new_active_mask, :] = x[new_active_mask]              # Save the updated positions
        densities[k+1, new_active_mask] = dens_sub[still_active_mask]     # Save the new densities
        bfields[k+1, new_active_mask] = bfield[still_active_mask]   # Save the new B-fields

        # ---------------------------------------------------------

        if np.all(un_masked):
            print("All values are False: means all density < 10^2")
            break
        mask = new_active_mask

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

        active_x = x[mask_rev]

        # Perform Heun step and update values
        _, bfield, dens_sub, vol, ke, pressure = Heun_step(
            active_x, -1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
        )
        
        pressure *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens_sub *= gr_cm3_to_nuclei_cm3

        # Calculate distances and update arrays
        distance_traveled = np.linalg.norm(active_x - x[mask_rev], axis=1) * pc_to_cm

        act_ind_rev = np.where(mask_rev.reshape(m,d))
        # Update distance and number density arrays directly
        distance_bck[k][act_ind_rev] = distance_traveled
        number_density_bck[k][act_ind_rev] = dens_sub

        still_active_mask_rev = dens_sub > 100
        
        #vol[un_masked_rev] = 0
        print(x[0], np.log10(dens[0]))

        non_zero_rev = vol > 0
        if len(vol[non_zero_rev]) == 0:
            break
        
        active_vol = vol[still_active_mask_rev]


        
        # Only consider active volumes that are non-zero
        volumes_to_minimize_rev = active_vol[active_vol > 0] 

        if len(volumes_to_minimize_rev) == 0:
            # If there are no active, non-zero volumes left, break the loop 
            # as there are no rays to propagate further based on this step size.
            break 
        dx_vec = np.min(((4 / 3) * volumes_to_minimize_rev / np.pi) ** (1 / 3))  # Increment step size

        threshold_rev += mask_rev.astype(int)  # Increment threshold count only for values still above 100

        active_incides_rev = np.where(mask_rev)[0]
        still_active_indices_rev = active_incides_rev[still_active_mask_rev]
        new_active_mask_rev = np.zeros_like(mask_rev, dtype = bool)
        new_active_mask_rev[still_active_indices_rev] = True
     
        x[new_active_mask_rev] -= dx_vec * directions[new_active_mask_rev]
  

        line_rev[k+1,:,:]      = line_rev[k,:,:]
        densities_rev[k+1,:]   = densities_rev[k,:]
        bfields_rev[k+1,:]     = bfields_rev[k,:]

        line_rev[k+1, new_active_mask_rev, :] = x[new_active_mask_rev]              # Save the updated positions
        densities_rev[k+1, new_active_mask_rev] = dens_sub[still_active_mask_rev]     # Save the new densities
        bfields_rev[k+1, new_active_mask_rev] = bfield[still_active_mask_rev]   # Save the new B-fields


        if np.all(un_masked_rev):
            print("All values are False: means all density < 10^2")
            break
        mask_rev = new_active_mask_rev
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

    numb_densities = numb_densities.reshape(4001, m, d)

    trajectory = np.zeros_like(numb_densities)
    column = np.zeros_like(numb_densities)

    print("Surviving lines: ", m, "out of: ", max_cycles)

    for _n in range(min(radius_vector.shape[1], trajectory.shape[1])):  # Iterate over the first dimension
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
    # Combine forward and backward arrays
    total_distance_LOS = np.concatenate((distance_bck[::-1], distance_fwd), axis=0)
    total_number_density_LOS = np.concatenate((number_density_bck[::-1], number_density_fwd), axis=0)
    return radius_vector, trajectory, numb_densities, [threshold, threshold_rev], column, total_distance_LOS, total_number_density_LOS




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


    # Initialize arrays to store distance traveled and number density at each step
    distance_fwd = np.zeros((N, max_cycles))
    number_density_fwd = np.zeros((N, max_cycles))
    distance_bck = np.zeros((N, max_cycles))
    number_density_bck = np.zeros((N, max_cycles))

    posBheun_fwd = np.zeros((N, max_cycles, 3))
    posBheun_bck = np.zeros((N, max_cycles, 3))
    ndBheun_fwd = np.zeros((N, max_cycles))
    ndBheun_bck = np.zeros((N, max_cycles))


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
            
            print('shape of active_fwd:', active_fwd.shape)
            print('shape of mask_fwd:', mask_fwd.shape)
            print('shape of local_densities_fwd:', local_densities_fwd.shape)
            print('shape of pos at index i:', posBheun_fwd[i].shape)
            print('shape of nd at index i:', ndBheun_fwd[i].shape)
            print('active mask', new_mask_fwd.shape)
            
            posBheun_fwd[i,mask_fwd] = active_fwd
            ndBheun_fwd[i,mask_fwd] = local_densities_fwd
            
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

            posBheun_bck[i,mask_bck] = active_bck
            ndBheun_bck[i,mask_bck] = local_densities_bck
            
            mask_bck[mask_bck] = new_mask_bck

            
        # if both masks false
        if not np.any(mask_fwd) and not np.any(mask_bck):
            break

           
    BDtotal_heun = column_fwd + column_bck
    return BDtotal_heun, posBheun_fwd, ndBheun_fwd, posBheun_bck, ndBheun_bck

def get_B_field_column_density_euler(
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


    posBeuler_fwd = np.zeros((N, max_cycles, 3))
    posBeuler_bck = np.zeros((N, max_cycles, 3))
    ndBeuler_fwd = np.zeros((N, max_cycles))
    ndBeuler_bck = np.zeros((N, max_cycles))


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
            
            next_fwd, _, _, _ = Euler_step(
                active_fwd, np.ones(len(active_fwd)) * 0.5, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
            )

            distance_traveled_fwd = np.linalg.norm(next_fwd - active_fwd, axis=1) * pc_to_cm


            column_fwd[mask_fwd] += local_densities_fwd * distance_traveled_fwd
            current_fwd[mask_fwd] = next_fwd

            posBeuler_fwd[i,mask_fwd] = active_fwd
            ndBeuler_fwd[i,mask_fwd] = local_densities_fwd

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
            
        
            next_bck, _, _, _ = Euler_step(
                active_bck, np.ones(len(active_bck)) * -0.5, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
            )
            
            distance_traveled_bck = np.linalg.norm(next_bck - active_bck, axis=1) * pc_to_cm

           
            column_bck[mask_bck] += local_densities_bck * distance_traveled_bck
            current_bck[mask_bck] = next_bck

            posBeuler_bck[i,mask_bck] = active_bck
            ndBeuler_bck[i,mask_bck] = local_densities_bck
            
            mask_bck[mask_bck] = new_mask_bck


            
        # if both masks false
        if not np.any(mask_fwd) and not np.any(mask_bck):
            break

    BDtotal_euler = column_fwd + column_bck
    return BDtotal_euler, posBeuler_fwd, ndBeuler_fwd, posBeuler_bck, ndBeuler_bck


def Debug(m,d,case,snap,seed,i,df):
    import pandas as pd
    
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
    if not os.path.exists(full_data_path):
        print("File not found:", full_data_path)
        one_cloud_N(i)
        return Debug(m,d,case,snap,seed,i, df)
    
    data = np.load(full_data_path)

    print(data.files)

    x_init  = data['x_init_points']
    mean_CD = data['mean_column_densities']
    path_CD_heun = data["pathcolumn_heun"]
    path_CD_euler = data["pathcolumn_euler"]
    distance_heun = data["distanceB_heun"]
    distance_euler = data["distanceB_euler"]
    number_density_heun = data["numberdensity_heun"]
    number_density_euler = data["numberdensity_euler"]


    final_column_density = mean_CD[-1, :]
    radial_distance_pc = np.linalg.norm(x_init, axis=1)
    radial_distance_cm = radial_distance_pc * pc_to_cm

    print("x_init points:", x_init)
    print("Final Column Densities:", final_column_density)
    print("Radial Distances (cm):", radial_distance_cm)
    print('shape of distance heun:', distance_heun.shape)
    print('shape of number density heun:', number_density_heun.shape)
    print('shape of distance euler:', distance_euler.shape)
    print('shape of number density euler:', number_density_euler.shape)
    print('shape x_init:', x_init.shape)

def NvsR(m,d,case,snap,seed,i, r_values_los, r_values_path, df):
    import pandas as pd
    from scipy.stats import linregress

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
    if not os.path.exists(full_data_path):
        print("File not found:", full_data_path)
        one_cloud_N(i)
        return NvsR(m,d,case,snap,seed,i, r_values_los, r_values_path, df)
    
    data = np.load(full_data_path)

    print(data.files)
    output_folder1 = "./graphs/rvsNheun"
    png_name1 = f"ColumnDensity_vs_RadialDistance_Heun_{seed}_{m}_{d}_{i}.png"

    output_folder2 = "./graphs/rvsNeuler"
    png_name2 = f"ColumnDensity_vs_RadialDistance_Euler_{seed}_{m}_{d}_{i}.png"

    x_init  = data['x_init_points']
    mean_CD = data['mean_column_densities']
    path_CD_heun = data["pathcolumn_heun"]
    path_CD_euler = data["pathcolumn_euler"]
    trajectories = data['trajectories']

    final_column_density = mean_CD[-1, :]
    radial_distance_pc = np.linalg.norm(x_init, axis=1)
    radial_distance_cm = radial_distance_pc * pc_to_cm


    #filter LOS data to keep negatives out (all N >0 ut a good practice to have :) )
    los_x  = radial_distance_cm[final_column_density > 0]
    los_y  = final_column_density[final_column_density > 0]
    #now for the B field path of Heun method
    path_x_heun = radial_distance_cm[path_CD_heun > 0]
    path_y_heun = path_CD_heun[path_CD_heun > 0]
    #the same for the euler method
    path_x_euler = radial_distance_cm[path_CD_euler > 0]
    path_y_euler = path_CD_euler[path_CD_euler > 0]
        
    los_slope, los_intercept, los_r, los_p, los_std_err = linregress(np.log10(los_x), np.log10(los_y))
    los_fitline = los_slope * np.log10(los_x) + los_intercept

    path_slope_heun, path_intercept_heun, path_r_heun, path_p_heun, path_std_err_heun = linregress(np.log10(path_x_heun), np.log10(path_y_heun))
    path_fitline_heun = path_slope_heun * np.log10(path_x_heun) + path_intercept_heun

    path_slope_euler, path_interncept_euler, path_r_euler, path_p_euler, path_std_err_euler = linregress(np.log10(path_x_euler), np.log10(path_y_euler))
    path_fitline_euler = path_slope_euler * np.log10(path_x_euler) + path_interncept_euler
        
    r_values_los.append(los_r)
    r_values_path.append(path_r_heun)
        
    # Plotting for Heun method vs radial distance
    plt.figure(figsize=(10, 6))

    plt.scatter(radial_distance_cm, final_column_density, s=10, alpha=0.5, color = "blue", label = " mean final column density along LOS")
    plt.scatter(radial_distance_cm, path_CD_heun, s = 10, alpha =0.3, color= "red", label = "column density along a B field path")

    plt.plot(los_x, 10**los_fitline, color="black", ls = ":", lw = 1, label=f"LOS Fit: $N \\propto r$ ($R^2={los_r**2:.2f}$)")
    plt.plot(path_x_heun, 10**path_fitline_heun, color = "black", ls = "--", lw = 1, label=f"Path Fit: $N \\propto r$ ($R^2={path_r_heun**2:.2f}$)")

    plt.legend()

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('$cm$', fontsize=14)
    plt.ylabel('$N (cm^{-2})$', fontsize=14)
        
    title_text = f"Column Density vs. Distance: Cloud {int(cloud_number)}"
    plt.title(title_text, fontsize=16)

    plt.grid(True, which="both", ls=":")
    plt.ylim( (10e20, 10e27) )

    info_text = (
    f"Center Coordinates: ({x:.2f}, {y:.2f}, {z:.2f})\n"
    f"Peak Density: {peak_density:.2f}"
    )
    plt.figtext(0.15, 0.02, info_text, fontsize=10, ha='left')

    full_path1 = os.path.join(output_folder1, png_name1)
    plt.savefig(full_path1, dpi=300)
    print("plot saved to: ", full_path1)

    # Plotting for Euler method vs radial distance
    plt.figure(figsize=(10, 6))
    plt.scatter(radial_distance_cm, final_column_density, s=10, alpha=0.5, color = "blue", label = " mean final column density along LOS")
    plt.scatter(radial_distance_cm, path_CD_euler, s = 10, alpha =0.3, color= "green", label = "column density along a B field path")
    plt.plot(los_x, 10**los_fitline, color="black", ls = ":", lw = 1, label=f"LOS Fit: $N \\propto r$ ($R^2={los_r**2:.2f}$)")
    plt.plot(path_x_euler, 10**path_fitline_euler, color = "black", ls = "--", lw = 1, label=f"Path Fit: $N \\propto r$ ($R^2={path_r_euler**2:.2f}$)")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$cm$', fontsize=14)
    plt.ylabel('$N (cm^{-2})$', fontsize=14)
    title_text = f"Column Density vs. Distance: Cloud {int(cloud_number)}"
    plt.title(title_text, fontsize=16)
    plt.grid(True, which="both", ls=":")
    plt.ylim( (10e20, 10e27) )
    info_text = (
    f"Center Coordinates: ({x:.2f}, {y:.2f}, {z:.2f})\n"
    f"Peak Density: {peak_density:.2f}"
    )
    plt.figtext(0.15, 0.02, info_text, fontsize=10, ha='left')

    full_path2 = os.path.join(output_folder2, png_name2)
    plt.savefig(full_path2, dpi=300)
    print("plot saved to: ", full_path2)


def BeulerheunvsR(m,d,case,snap,seed,i,df):
    import pandas as pd

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
    if not os.path.exists(full_data_path):
        print("File not found:", full_data_path)
        one_cloud_N(i)
        return BeulerheunvsR(m,d,case,snap,seed,i,df)
    data = np.load(full_data_path)


    x_init  = data['x_init_points']
    mean_CD = data['mean_column_densities']
    path_CD_heun = data["pathcolumn_heun"]
    path_CD_euler = data["pathcolumn_euler"]
    trajectories = data['trajectories']

    posBheun_fwd = data["posBheun_fwd"]
    ndBheun_fwd  = data["ndBheun_fwd"]
    posBheun_bck = data["posBheun_bck"]
    ndBheun_bck  = data["ndBheun_bck"]

    posBeuler_fwd = data["posBeuler_fwd"]
    ndBeuler_fwd  = data["ndBeuler_fwd"]
    posBeuler_bck = data["posBeuler_bck"]
    ndBeuler_bck  = data["ndBeuler_bck"]

    final_column_density = mean_CD[-1, :]
    radial_distance_pc = np.linalg.norm(x_init, axis=1)
    radial_distance_cm = radial_distance_pc * pc_to_cm



    full_positions_heun = np.concatenate((posBheun_bck[::-1,:,:], posBheun_fwd[1:,:,:]), axis=0)
    full_densities_heun = np.concatenate((ndBheun_bck[::-1,:], ndBheun_fwd[1:,:]), axis=0)
    num_steps = full_positions_heun.shape[0]
    m = full_positions_heun.shape[1]

    centered_positions = np.zeros((num_steps, m, 3))

    arr_total_distance_heun = np.zeros((num_steps, m))
    print("Number of steps along B field lines (Heun): ", num_steps)
    print("Number of lines along B field lines (Heun): ", m)
    print("shape of full positions (Heun): ", full_positions_heun.shape)
    print("shape of full densities (Heun): ", full_densities_heun.shape)
    plt.figure()
    def cent_positions(full_positions_heun):
        for k in range(m):
            centered_positions[:, k, :] = full_positions_heun[:, k, :] - full_positions_heun[0, k, :]
        return centered_positions
    
    
    centered_positions = cent_positions(full_positions_heun)
    total_distance_heun = np.zeros(m)
    for i in range(1, num_steps):
    
        step_distance = np.linalg.norm(centered_positions[i, :, :] - centered_positions[i-1, :, :], axis=1)
        total_distance_heun += step_distance
        arr_total_distance_heun[i, :] = total_distance_heun

    arr_total_distance_heun_cm = arr_total_distance_heun * pc_to_cm
 

    full_positions_euler = np.concatenate((posBeuler_bck[::-1,:,:], posBeuler_fwd[1:,:,:]), axis=0)
    full_densities_euler = np.concatenate((ndBeuler_bck[::-1,:], ndBeuler_fwd[1:,:]), axis=0)
    num_steps_euler = full_positions_euler.shape[0]
    m_euler = full_positions_euler.shape[1]
    print("Number of steps along B field lines (Euler): ", num_steps_euler)
    print("Number of lines along B field lines (Euler): ", m_euler)
    print("shape of full positions (Euler): ", full_positions_euler.shape)
    print("shape of full densities (Euler): ", full_densities_euler.shape)

    centered_positions_euler = np.zeros((num_steps_euler, m_euler, 3))
    def cent_positions_euler(full_positions_euler):
        for k in range(m_euler):
            centered_positions_euler[:, k, :] = full_positions_euler[:, k, :] - full_positions_euler[0, k, :]
        return centered_positions_euler
    centered_positions_euler = cent_positions_euler(full_positions_euler)
    arr_total_distances_euler = np.zeros((num_steps_euler, m_euler))

    total_distance_euler = np.zeros(m_euler)
    for j in range(1, num_steps_euler):
        
        step_distance_euler = np.linalg.norm(centered_positions_euler[j, :, :] - centered_positions_euler[j-1, :, :], axis=1)
        total_distance_euler += step_distance_euler
        arr_total_distances_euler[j, :] = total_distance_euler
    
    arr_total_distances_euler_cm = arr_total_distances_euler * pc_to_cm

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    #indices of the lines that get the highest and lowest column densities
    mask = np.ones(m, dtype=bool)
    mask_euler = np.ones(m_euler, dtype=bool)
    for n in range(m):
        if all(full_densities_heun[:,n]) < 1000:
            mask[n] = False
    for n in range(m_euler):
        if all(full_densities_euler[:,n]) < 1000:
            mask_euler[n] = False
    
    masked_arr_total_distance_heun_cm = arr_total_distance_heun_cm[:, mask]
    masked_full_densities_heun = full_densities_heun[:, mask]
    masked_arr_total_distances_euler_cm = arr_total_distances_euler_cm[:, mask_euler]
    masked_full_densities_euler = full_densities_euler[:, mask_euler]

    print('Want to print all lines? (y/n)')
    ans = input()

    ax1.set_title(f"number density along B Field Lines (Heun) - Cloud {int(cloud_number)}")
    ax1.set_xlabel("Distance along B Field Line (cm)")
    ax1.set_ylabel("Number Density (cm$^{-3}$)")
    ax2.set_title(f"number density along B Field Lines (Euler) - Cloud {int(cloud_number)}")
    ax2.set_xlabel("Distance along B Field Line (cm)")
    ax2.set_ylabel("Number Density (cm$^{-3}$)")

    ax1.grid(True, which="both", ls=":")
    ax2.grid(True, which="both", ls=":")


    if ans.lower() == 'y':
        for k in range(1, m):  # Plot every 5th line for clarity
            ax1.plot(arr_total_distance_heun_cm[:, k], full_densities_heun[:, k], alpha=0.3)
        for k in range(0, m_euler-1, 100):  # Plot every 5th line for clarity
            ax2.plot(arr_total_distances_euler_cm[:, k], full_densities_euler[:, k], alpha=0.3)
    else:
        lineofchoice = int(input("Enter the line number you want to plot: "))
        ax1.plot(arr_total_distance_heun_cm[:, lineofchoice], full_densities_heun[:, lineofchoice])
        ax2.plot(arr_total_distances_euler_cm[:, lineofchoice], full_densities_euler[:, lineofchoice])
                 
    plt.tight_layout()
    plt.show()

    rerun = input(print('1)this graph again 2)stop'))
    if rerun == '1':
        BeulerheunvsR(m,d,case,snap,seed,i,df)
    if rerun == '2':
        return
 
def numberdensity(m,d,case, snap, seed, i, df):
    import pandas as pd

    cloud_data = df.iloc[i]
    cloud_number = cloud_data['index']
    peak_density = cloud_data['Peak_Density']
    x = cloud_data['CloudCord_X']
    y = cloud_data['CloudCord_Y']
    z = cloud_data['CloudCord_Z']

    data_coordinates = np.load
    data_directory = os.path.join("thesis_los", case, snap)
    data_name = f"DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{i}.npz"
    full_data_path = os.path.join(data_directory, data_name)
    if not os.path.exists(full_data_path):
        print("File not found:", full_data_path)
        one_cloud_N(i)
        return numberdensity(m,d,case,snap,seed,i,df)
    data = np.load(full_data_path)

    x_init  = data['x_init_points']
    mean_CD = data['mean_column_densities']
    path_CD_heun = data["pathcolumn_heun"]
    path_CD_euler = data["pathcolumn_euler"]
    trajectories = data['trajectories']

    posBheun_fwd = data["posBheun_fwd"]
    ndBheun_fwd  = data["ndBheun_fwd"]
    posBheun_bck = data["posBheun_bck"]
    ndBheun_bck  = data["ndBheun_bck"]

    posBeuler_fwd = data["posBeuler_fwd"]
    ndBeuler_fwd  = data["ndBeuler_fwd"]
    posBeuler_bck = data["posBeuler_bck"]
    ndBeuler_bck  = data["ndBeuler_bck"]

    trajectories = data["trajectories"]
    positions = data['positions']
    numberdens_los = data['numberdensity_LOS']

    full_positions_heun = np.concatenate((posBheun_bck[::-1,:,:], posBheun_fwd[1:,:,:]), axis=0)

    full_densities_heun = np.concatenate((ndBheun_bck[::-1,:], ndBheun_fwd[1:,:]), axis=0)


    num_steps = full_positions_heun.shape[0]
    m = full_positions_heun.shape[1]

    final_column_density = mean_CD[-1,:]
    ratio = final_column_density / path_CD_heun
    highest_ratio_index = np.argmax(ratio)

    centered_positions_b = np.zeros((num_steps, 3))
    centeres_positions_los = np.zeros((d, num_steps+2,3))

    point_maxratio = trajectories[:,highest_ratio_index,:]
    print(point_maxratio.shape)
    

    reshape_positions = positions.reshape((4001,m,d,3))
    
    positions_points = reshape_positions[:,highest_ratio_index,:,:]
    path_positiosn = full_positions_heun[:,highest_ratio_index,:]

    print(positions.shape)
    print(reshape_positions.shape)
    
    arr_total_distance_heun = np.zeros((num_steps))
    arr_total_distance_los = np.zeros((d,num_steps))

    print("Number of steps along B field lines (Heun): ", num_steps)
    print("Number of lines along B field lines (Heun): ", m)
    print("shape of full positions (Heun): ", full_positions_heun.shape)
    print("shape of full densities (Heun): ", full_densities_heun.shape)
    plt.figure()
    def cent_positions(path_positiosn, positions_points):
        
    
        centered_positions_b = path_positiosn - path_positiosn[0]

        i = 0
        while i < d:
            centeres_positions_los[i,:,:] = positions_points[:,i,:] - positions_points[0,i,:]
            i += 1
        return centered_positions_b, centeres_positions_los
    
    
    centered_positions_b, centered_positions_los = cent_positions(path_positiosn, positions_points)

    total_distance_heun = 0
    total_distance_los = 0
    for i in range(1, num_steps):
    
        step_distance_crs = np.linalg.norm(centered_positions_b[i, :] - centered_positions_b[i-1, :])
        total_distance_heun += step_distance_crs
        arr_total_distance_heun[i] = total_distance_heun

        j=0
        print(centered_positions_los.shape)
        while j < d:
            step_distance_los = np.linalg.norm(centered_positions_los[j,i,:] - centered_positions_los[j,i-1,:])
            total_distance_los += step_distance_los
            arr_total_distance_los[j,i] = total_distance_los
            j += 1


    arr_total_distance_heun_cm = arr_total_distance_heun * pc_to_cm
    arr_total_distance_los_cm = arr_total_distance_los * pc_to_cm
 
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), sharey=True)

    ax1.set_title(f"number density along B Field Lines (heun integration) - Cloud {int(cloud_number)}")
    ax1.set_xlabel("Distance along b FIELD lINE (cm)")
    ax1.set_ylabel("Number Density")
    ax2.set_title(f"number density along lines of sight - Cloud {int(cloud_number)}")
    ax2.set_xlabel("Distance along the LOS (cm)")
    ax2.set_ylabel("Number Density")

    ax1.grid(True, which="both", ls=":")
    ax2.grid(True, which="both", ls=":")

    densities_path = full_densities_heun[:,highest_ratio_index]
    print('densities_path shape: ', densities_path.shape)
    ax1.plot(arr_total_distance_heun_cm, densities_path)

    line  = 0
    print(numberdens_los.shape)
    print(arr_total_distance_los_cm.shape)

    print(full_densities_heun.shape)
    print(full_positions_heun.shape)

    print('lenght of arr_total_distance_heun ', arr_total_distance_heun_cm.shape)
    print("shape of full densities (Heun): ", full_densities_heun.shape)

    number_dens_los_corrected = numberdens_los[:-1,:,:]
    print(number_dens_los_corrected.shape)
    while line < d:
        ax2.plot(arr_total_distance_los_cm[line,:], number_dens_los_corrected[:,highest_ratio_index,line])
        line = line+1
    
    plt.tight_layout()
    plt.show()
#def BandLOSnvsR(m,d,case,snap,seed,i,df):
   #for this function we compare number densities of 
def graphs(m,d,case,snap,seed):
    folderrvsNheun = os.path.join('graphs', 'rvsNheun')
    if not os.path.exists(folderrvsNheun):
        os.makedirs(folderrvsNheun)

    folderrvsNeuler = os.path.join('graphs', 'rvsNeuler')
    if not os.path.exists(folderrvsNeuler):
        os.makedirs(folderrvsNeuler)

    foldercontour = os.path.join('graphs', 'contour_maps')
    if not os.path.exists(foldercontour):
        os.makedirs(foldercontour)

    foldernumberdensity = os.path.join('graphs', 'numberdensity')
    if not os.path.exists(foldernumberdensity):
        os.makedirs(foldernumberdensity)

    r_values_los  = []
    r_values_path = []

    coordirec = os.path.join("clouds")
    cornames  = f"{case}_clouds.txt"
    full_cord_path = os.path.join(coordirec, cornames)
    df = pd.read_csv(full_cord_path)

    
    gr = input('What do you want to plot?: (1) N vs R for Heun and Euler 2) Contour maps of number density around LOS (3) number density along B field lines for Heun and Euler methods (4) Debugging info (5) heatmap (6) number density')
    if gr == '1':
        allorone = input('Do you want to plot for (a) all clouds or (b) one cloud? ')
        if allorone.lower() == 'a':
            i = 0
            while i < num_clouds:
                NvsR(m,d,case,snap,seed,i, r_values_los, r_values_path, df)
                i += 1
        elif allorone.lower() == 'b':
                i = int(input("Enter the cloud number you want to plot: "))
                NvsR(m,d,case,snap,seed,i, r_values_los, r_values_path, df)
    elif gr == '2':
        allorone = input('Do you want to plot for (a) all clouds or (b) one cloud? ')
        if allorone.lower() == 'a':
            i = 0
            while i < num_clouds:
                plot_highest_ratio_projection(i, seed, m, d, new_folder)
                i += 1
        elif allorone.lower() == 'b':
            i = int(input("Enter the cloud number you want to plot: "))
        
    elif gr == '3':
        allorone = input('Do you want to plot for (a) all clouds or (b) one cloud? ')
        if allorone.lower() == 'a':
            i = 0
            while i < num_clouds:
                BeulerheunvsR(m,d,case,snap,seed,i)
                i += 1
        elif allorone.lower() == 'b':
            i = int(input("Enter the cloud number you want to plot: "))
            BeulerheunvsR(m,d,case,snap,seed,i,df)
    elif gr == '4':
        i = int(input("Enter the cloud number you want to plot debugging info for: "))
        Debug(m,d,case,snap,seed,i,df)
    elif gr == '5':
        allorone = input('Do you want to plot for (a) all clouds or (b) one cloud? ')
        if allorone.lower() == 'a':
            i = 0
            while i < num_clouds:
                PlaneOnALOS(m,d,case,snap,seed,i,df)
                i += 1
        elif allorone.lower() == 'b':
            i  = int(input('Enter the cloud number you want to plot: '))
            PlaneOnALOS(m,d,case,snap,seed,i,df)
    elif gr == '6':
        i = int(input("which cloud)"))
        numberdensity(m,d,case,snap,seed,i,df)

        
        
def PlaneOnALOS(m,d,case,snap,seed,i,df):
    #first, find which line of sight to use
    import pandas as pd
    from scipy.spatial import KDTree

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
    if not os.path.exists(full_data_path):
        print("File not found:", full_data_path)
        one_cloud_N(i)
        return PlaneOnALOS(m,d,case,snap,seed,i,df)
    data = np.load(full_data_path)


    x_init  = data['x_init_points']
    mean_CD = data['mean_column_densities']
    path_CD_heun = data["pathcolumn_heun"]
    path_CD_euler = data["pathcolumn_euler"]
    trajectories = data['trajectories']

    posBheun_fwd = data["posBheun_fwd"]
    ndBheun_fwd  = data["ndBheun_fwd"]
    posBheun_bck = data["posBheun_bck"]
    ndBheun_bck  = data["ndBheun_bck"]

    posBeuler_fwd = data["posBeuler_fwd"]
    ndBeuler_fwd  = data["ndBeuler_fwd"]
    posBeuler_bck = data["posBeuler_bck"]
    ndBeuler_bck  = data["ndBeuler_bck"]
    positions = data['positions']

    full_densities_los = data['numberdensity_LOS']


    cloud_center = np.array([x, y, z])
    Pos_copy = Pos.copy()
    VoronoiPos_copy = VoronoiPos.copy()
    
    Pos_copy -= cloud_center
    VoronoiPos_copy -= cloud_center

    for dim in range(3):  # Loop over x, y, z
        pos_from_center = Pos_copy[:, dim]
        boundary_mask = pos_from_center > Boxsize / 2
        Pos_copy[boundary_mask, dim] -= Boxsize
        VoronoiPos_copy[boundary_mask, dim] -= Boxsize
        
        boundary_mask = pos_from_center < -Boxsize / 2
        Pos_copy[boundary_mask, dim] += Boxsize
        VoronoiPos_copy[boundary_mask, dim] += Boxsize


    tree = KDTree(Pos_copy)

    print("shape of trajectories: ", trajectories.shape)
    print('Shape of the mean column densities', mean_CD.shape)
    print('Shape of Bheun arrya', path_CD_heun.shape)
    #find point with the highest ratio of NLOS vs NBpath

    final_column_density = mean_CD[-1, :]
    print('shape of inal_N', final_column_density.shape)
    ratio = final_column_density /path_CD_heun
    print(ratio.shape)

    highest_ratio_index = np.argmax(ratio)

    point_maxratio = trajectories[:,highest_ratio_index,:]
    print(point_maxratio.shape)

    reshape_positions = positions.reshape((4001,1000,20,3))
    positions_point = reshape_positions[:,highest_ratio_index,:,:]

    print(positions.shape)
    print(reshape_positions.shape)

       # final_column_density = mean_CD[-1,:]
    #ratio = final_column_density / path_CD_heun
    #highest_ratio_index = np.argmax(ratio)

    #check the line of sight with the highest number density on its path
    highest_number_density_index = 1
    for j in range(2, d):
        max_density_along_los = np.max(full_densities_los[:,highest_ratio_index,j])  # Assuming density is at index 2
        if max_density_along_los > np.max(full_densities_los[:,highest_ratio_index,highest_number_density_index]):
            highest_number_density_index = j
    print(f"Highest ratio LOS index: {highest_ratio_index}")
    print(f"Highest number density along LOS index: {highest_number_density_index}")


    # Get the full LOS trajectory (all points along the line of sight)
    los_trajectory = positions_point[:,highest_number_density_index,:]  # Using direction index 3, all points along LOS
    los_trajectory_length = los_trajectory.shape[0]
    
    # Get the x_init point for this LOS (starting point)
    x_init_point = x_init[highest_ratio_index]  # Already cloud-centered
    
    # Get the LOS direction vector (from start to end) for plane construction
    los_start = los_trajectory[0,:]  # First point
    los_end = los_trajectory[-1,:]   # Last point
    vectorlos = los_end - los_start
    los_length = np.linalg.norm(vectorlos)
    unit_los = vectorlos / los_length if los_length > 0 else np.array([1,0,0])
    
    print(f"LOS trajectory has {los_trajectory_length} points")
    print(f"x_init point (cloud-centered): {x_init_point}")
    print(f"LOS direction: {unit_los}")
    print(f"LOS length: {los_length:.3f} pc")
    
    # Choose axis to be perpendicular to (default: z-axis for xy plane)
    # User can change this to [1,0,0] for yz plane or [0,1,0] for xz plane
    axis_normal = np.array([0, 0, 1]) 
    
    # Calculate two perpendicular vectors in the plane
    # First, get a vector perpendicular to both the axis and LOS
    if np.abs(np.dot(unit_los, axis_normal)) > 0.99:
        # if LOS is nearly parallel to axis, use a different reference
        if np.abs(axis_normal[2]) > 0.9:
            ref_vec = np.array([1, 0, 0])  # Use x-axis as reference
        else:
            ref_vec = np.array([0, 0, 1])  # Use z-axis as reference
    else:
        ref_vec = axis_normal
    
    perp1 = np.cross(axis_normal, unit_los)
    perp1_norm = np.linalg.norm(perp1)
    if perp1_norm < 1e-10:
        # If LOS is parallel to axis, use a different approach
        perp1 = np.cross(ref_vec, unit_los)
        perp1_norm = np.linalg.norm(perp1)
    perp1 = perp1 / perp1_norm if perp1_norm > 1e-10 else np.array([1, 0, 0])
    
    perp2 = np.cross(unit_los, perp1)
    perp2_norm = np.linalg.norm(perp2)
    perp2 = perp2 / perp2_norm if perp2_norm > 1e-10 else np.array([0, 1, 0])
    
    print(f"Perpendicular vector 1: {perp1}")
    print(f"Perpendicular vector 2: {perp2}")
    print(f"Dot product (should be ~0): {np.dot(perp1, perp2):.6e}")
    print(f"Dot product with LOS (should be ~0): {np.dot(perp1, unit_los):.6e}")

    # Determine appropriate grid size based on cloud scale
    # The LOS extends until density drops below densthresh (100 cm^-3), 
    
    # Use LOS length 
    grid_size_pc = max(los_length * 2, 5)  # At least 10 pc, or 2x LOS length for better coverage
    grid_size_pc = min(grid_size_pc, 10.0)  # Cap at 30 pc to avoid excessive computation
    
    Grid_resolution = 10000

    perpfr = [-unit_los[1], unit_los[0], 0]
    perpfr_norm = np.linalg.norm(perpfr)
    
    perpunitfr = perpfr / perpfr_norm 
    
    print(f"LOS length: {los_length:.3f} pc")
    print(f"Grid size: {grid_size_pc:.3f} pc (2x LOS length, bounded 10-30 pc), Resolution: {Grid_resolution}")

    A1d = np.linspace(-grid_size_pc, grid_size_pc, Grid_resolution)
    B1d = np.linspace(-grid_size_pc, grid_size_pc, Grid_resolution)

    #create the mesh 2d
    A_grid, B_grid = np.meshgrid(A1d, B1d)

    # use the x_init point as the origin of the plane
    los_point = x_init_point  # This is already cloud-centered
    
    # Build the plane using the two perpendicular vectors
    A_component = A_grid[:,:, np.newaxis] * unit_los
    B_component = B_grid[:,:, np.newaxis] * perpunitfr

    Q_3d = los_point + A_component + B_component

    Q_query_points = Q_3d.reshape(-1,3)

    print(f"Query points range: X=[{Q_query_points[:,0].min():.3f}, {Q_query_points[:,0].max():.3f}] pc")
    print(f"Query points range: Y=[{Q_query_points[:,1].min():.3f}, {Q_query_points[:,1].max():.3f}] pc")
    print(f"Query points range: Z=[{Q_query_points[:,2].min():.3f}, {Q_query_points[:,2].max():.3f}] pc")

    distances, indices = tree.query(Q_query_points, k=1)

    # Get densities at query points (pc)
    _, _, dens, _ = find_points_and_get_fields(
       Q_query_points, Bfield, Density, Density_grad, Pos_copy, VoronoiPos_copy
    )

    print(f"Raw Density Range: [{dens.min():.3e}, {dens.max():.3e}] g/cm³")
    dens *= gr_cm3_to_nuclei_cm3 
    print(f"Scaled Density Range: [{dens.min():.3e}, {dens.max():.3e}] cm⁻³")

    dens_grid = dens.reshape(Grid_resolution, Grid_resolution)

    masked_data = ma.masked_where(dens_grid<100, dens_grid)


    # Check which contour levels are actually present in the data
    contour_levels = [100]  # cm^-3
    #valid_levels = [level for level in contour_levels if dens.min() <= level <= dens.max()]
    #if len(valid_levels) < len(contour_levels):
    #    print(f"Warning: Some contour levels not in data range. Using: {valid_levels}")
    #    if len(valid_levels) == 0:
    #        # If no levels are valid, use some levels within the range
     #       valid_levels = np.logspace(np.log10(max(dens.min(), 1)), np.log10(dens.max()), 5)
     #       print(f"Using auto-generated levels: {valid_levels}")
    
    # For pcolormesh, we need to handle the grid properly
    # Create coordinate arrays for pcolormesh (one element larger)
    A_edges = np.linspace(-grid_size_pc, grid_size_pc, Grid_resolution + 1)
    B_edges = np.linspace(-grid_size_pc, grid_size_pc, Grid_resolution + 1)
    A_grid_edges, B_grid_edges = np.meshgrid(A_edges, B_edges)

    cmap = cm.viridis.copy()
    cmap.set_bad(color='white', alpha=0)
    
    plt.figure(figsize=(12,10))

    im = plt.pcolormesh(A_grid_edges, B_grid_edges, masked_data, shading='auto', cmap=cmap)
    plt.colorbar(im, label='Number Density (cm⁻³)')
    
    # Add contour lines at specified densities
    #if len(valid_levels) > 0:
    #    contours = plt.contour(A_grid, B_grid, dens_grid, levels=valid_levels, 
    #                          colors='white', linewidths=1.0, alpha=0.9)
    #   plt.clabel(contours, inline=True, fontsize=11, fmt='%d cm⁻³')
    # Add contour lines at specified densities
   

    
    # Mark the x_init point (which is at the origin of the plane coordinates)
    plt.plot(0.0, 0.0, 'ro', markersize=10, label='x_init point', zorder=5, markeredgecolor='white', markeredgewidth=2)
    
    plt.xlabel('x (pc)', fontsize=12)
    plt.ylabel('y (pc)', fontsize=12)
    plt.title(f'Number Density Map on Plane - Cloud {int(cloud_number)}', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.axis('equal')  # Make aspect ratio equal for better visualization
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

    



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

    def column_density_LOS_Bfield_clouds():
        full_CD_calculations_time_average = np.zeros((2,num_clouds))
        full_CD_calculations_time         = np.zeros((2,))
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

            start_time_heun = time.perf_counter()
            BD_total_heun = get_B_field_column_density(x_init,Bfield,Density,densthresh,N,max_cycles,Density_grad, Volume, VoronoiPos = VoronoiPos_copy, Pos = Pos_copy)
            end_time_heun = time.perf_counter()

            full_time_heun_CDs = end_time_heun - start_time_heun
            
            start_time_euler = time.perf_counter()
            BD_total_euler = get_B_field_column_density_euler(x_init,Bfield,Density,densthresh,N,max_cycles,Density_grad, Volume, VoronoiPos = VoronoiPos_copy, Pos = Pos_copy)
            end_time_euler = time.perf_counter()

            full_time_euler_CDs = end_time_euler - start_time_euler

            full_CD_calculations_time[0,i] = full_time_heun_CDs
            full_CD_calculations_time[1,i] = full_time_euler_CDs


            column_reshaped = column.reshape(column.shape[0],m,d) #separates the column densities per point, per directions
            mean_column_per_point = np.mean(column_reshaped, axis= 2) #takes the mean over the directions 
            np.savez(os.path.join(new_folder, f"DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{i}.npz"),
                trajectories          = trajectory,
                densities             = numb_densities,
                positions             = radius_vector, 
                mean_column_densities = mean_column_per_point,
                x_init_points         = x_init,
                snapshot_number       = int(num_file),
                pathcolumn_heun       = BD_total_heun,
                pathcolumn_euler      = BD_total_euler,
                full_columns          = column_reshaped,
                CD_times              = full_CD_calculations_time,
                )
            
        average_time_heun = np.mean(full_CD_calculations_time, axis=0)
        average_time_euler = np.mean(full_CD_calculations_time, axis=1) 

        print("Average time for full CD calculations Heun: ", average_time_heun)
        print("Average time for full CD calculations Euler: ", average_time_euler)

    def one_cloud_N(i):
        
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

        radius_vector, trajectory, numb_densities, th, column, total_distance_LOS, total_number_density_LOS = get_line_of_sight(x_init, directions, Pos=Pos_copy, VoronoiPos=VoronoiPos_copy)
        threshold, threshold_rev = th

        start_time_heun = time.perf_counter()
        BD_total_heun, posBheun_fwd, ndBheun_fwd, posBheun_bck, ndBheun_bck = get_B_field_column_density(x_init,Bfield,Density,densthresh,N,max_cycles,Density_grad, Volume, VoronoiPos = VoronoiPos_copy, Pos = Pos_copy)
        end_time_heun = time.perf_counter()

        full_time_heun_CDs = end_time_heun - start_time_heun
            
        start_time_euler = time.perf_counter()
        BD_total_euler, posBeuler_fwd, ndBeuler_fwd, posBeuler_bck, ndBeuler_bck = get_B_field_column_density_euler(x_init,Bfield,Density,densthresh,N,max_cycles,Density_grad, Volume, VoronoiPos = VoronoiPos_copy, Pos = Pos_copy)
        end_time_euler = time.perf_counter()

        full_time_euler_CDs = end_time_euler - start_time_euler

        print(posBheun_fwd.shape)
        print(posBeuler_fwd.shape)

        


        column_reshaped = column.reshape(column.shape[0],m,d) #separates the column densities per point, per directions
        mean_column_per_point = np.mean(column_reshaped, axis= 2) #takes the mean over the directions 
        np.savez(os.path.join(new_folder, f"DataBundle_MeanCD_andpathD_{seed}_{m}_{d}_{i}.npz"),
            trajectories          = trajectory,
            densities             = numb_densities,
            positions             = radius_vector, 
            mean_column_densities = mean_column_per_point,
            x_init_points         = x_init,
            snapshot_number       = int(num_file),
            pathcolumn_heun       = BD_total_heun,
            pathcolumn_euler      = BD_total_euler,
            full_columns          = column_reshaped,
            distance_LOS         = total_distance_LOS,
            numberdensity_LOS     = total_number_density_LOS,
            posBheun_fwd         = posBheun_fwd,
            ndBheun_fwd          = ndBheun_fwd,
            posBheun_bck         = posBheun_bck,
            ndBheun_bck          = ndBheun_bck,
            posBeuler_fwd        = posBeuler_fwd,
            ndBeuler_fwd         = ndBeuler_fwd,
            posBeuler_bck        = posBeuler_bck,
            ndBeuler_bck         = ndBeuler_bck,
            )
            
    inp = input("What do you want to compute? Press N for column densities and G for graphs: ") #NorG
    if inp == 'N':
        all_clouds = input("Do you want to compute for all clouds or a single one? Press A for all or S for single: ") #AorS
        if all_clouds == 'S':
            i = int(input("Enter the cloud number you want to compute: "))
            one_cloud_N(i)
        elif all_clouds == 'A':
            i = 0
            while i < num_clouds:
                one_cloud_N(i)
                i += 1
    elif inp == 'G':
        m     = int(input("Enter number of points: "))
        d     = int(input("Enter Number of lines of sight per point: "))
        case  = input("Enter case (ideal/amb): ")
        snap  = input("Enter snapshot number: ")
        seed  = int(input("Enter seed number: "))

        graphs(m,d,case,snap,seed)


#this version tries to unify all of my scripts into one for better analysis and less redundancy