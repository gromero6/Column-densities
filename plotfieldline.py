import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
import math
from load_data import Codedata
from snap_data import snap_data
from library import find_points_and_get_fields, Heun_step, Euler_step
from library import gr_cm3_to_nuclei_cm3, pc_to_cm
from library import velocity_unit, mass_unit, length_unit

def wrap_positions(Pos, VoronoiPos, Boxsize):
    for dim in range(3):
        mask = Pos[:, dim] > Boxsize / 2
        Pos[mask, dim] -= Boxsize
        VoronoiPos[mask, dim] -= Boxsize
        mask = Pos[:, dim] < -Boxsize / 2
        Pos[mask, dim] += Boxsize
        VoronoiPos[mask, dim] += Boxsize
    return Pos, VoronoiPos

def get_cloud_center(case, cloudnum):
    cloud_dir = os.path.join("clouds", f"{case}_clouds.txt")
    df = pd.read_csv(cloud_dir)
    cloud_data = df.iloc[cloudnum]
    return np.array([
        cloud_data['CloudCord_X'],
        cloud_data['CloudCord_Y'],
        cloud_data['CloudCord_Z']
    ])

def plot_field_line(dist_full, absb_full, pos_full, number_density_full, simulation, cloud_center, m_use, cloudnum, case, snapshot, seed):
    VoronoiPos_cloud = simulation.VoronoiPos - cloud_center
    VoronoiPos_cloud, _ = wrap_positions(VoronoiPos_cloud, VoronoiPos_cloud.copy(), simulation.Boxsize)

    #Convert B to microGauss
    unit_B = velocity_unit * np.sqrt(4 * np.pi * mass_unit / length_unit**3)
    B_microG = absb_full * unit_B * 1e6

    tree = cKDTree(VoronoiPos_cloud)
    weird_mask = np.zeros(m_use, dtype=bool)
    maurotraj = np.load('./ArepoTrajectory3.npy')
    maurodens = np.load('./ArepoNumberDensities3.npy')
    maurofield = np.load('./ArepoMagneticFields3.npy')
    mauropos = np.load('./ArePositions3.npy')

    print(maurotraj.shape, "trajectory shape")
    print(maurodens.shape, "number density shape")
    print(maurofield.shape, "magnetic field shape")
    print(mauropos.shape, 'radius vector shape')
    

    print(maurodens)

    slicing_fwd = 0
    for i in maurodens:
        if (i <= 100):
            slicing_fwd += slicing_fwd
        else: 
            break

    print('slice fwd:', slicing_fwd)
    #print('slice bck', slicing_bck)

    maurotraj = maurotraj[slicing_fwd:]
    #maurotraj = maurotraj[:-slicing_bck]

    maurofield = maurofield[slicing_fwd:]
    #maurofield = maurofield[:-slicing_bck]
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8)) 


    

    for idx in range(m_use):

        mauroinverse = maurofield /( (1.99e+33/(3.086e+18*100_000.0))**(-1/2))
        mauromicrogauss = mauroinverse  * unit_B * 1e6 

        pts = pos_full[:, idx, :]
        pts = pts[1:]
        pts = pts[:-1]
        B_microG = B_microG[1:]
        B_microG = B_microG[:-1]
        dist_pc = np.zeros(len(pts))


        for j in range(1, len(pts)):
            step = np.linalg.norm(pts[j] - pts[j-1])
            dist_pc[j] = dist_pc[j-1] + step
        x0 = [-0.23098224497499614261, 0.17261036189460871038, 0.051270637917016421981]
        subs = x0 - pts[0,:]
        distance = np.linalg.norm(subs)
        print(pts.shape)
        print(B_microG.shape)
        xaxis = dist_pc-distance
        xaxis2= (maurotraj * 1/pc_to_cm)-0.84552 - distance
        ax1.plot(xaxis, B_microG[:,idx], color='steelblue', alpha=0.5, linewidth=0.8)
        ax1.plot(xaxis2, mauromicrogauss * 78571*1.02, color='red', alpha=0.5, linewidth=0.8)
        #ax1.plot((maurotraj * 1/pc_to_cm)-0.9, mauromicrogauss * 78571*1.02, color='red', alpha=0.5, linewidth=0.8)

    print(mauromicrogauss.shape)
    print(maurotraj.shape)
    ax1.legend()
    ax1.set_xlabel("Distance along field line (pc)")
    ax1.set_ylabel(r"$|B|$ ($\mu$G)")
    ax1.grid(True, ls=':', which='both')
    ax1.set_xlim(min(xaxis), max(xaxis))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    case = 'ideal'
    snapshot = '430'
    cloudnum = 0
    seed = '12345'
    m_use = 1
    N_max = 50000
    density_threshold = 100.0

    integrationmethod = input("euler or heun?").lower()
    if integrationmethod == 'euler':
        intmeth = Euler_step
    if integrationmethod == 'heun':
        intmeth = Heun_step
    plot_func = plot_field_line
    out_path = os.path.join("thesis_los", case, snapshot, f"FieldLines_cloud{cloudnum}_m{m_use}_seed{seed}_{integrationmethod}.npz")

    
    try:
        simulation = snap_data(snapshot, case, str(cloudnum))
        cloud_center = get_cloud_center(case, cloudnum)
    except NameError:
        print("Required external functions (snap_data, get_cloud_center) are not defined. Cannot run integration.")
        exit()


    if os.path.exists(out_path):
        print(f"Loading existing data: {out_path}")
        data = np.load(out_path)
        pos_full = data["positions"]
        absb_full = data["absB"]
        dist_full = data["distance_cm"]
        number_density = data['number_density'] 


        plot_func(dist_full, absb_full, pos_full, number_density, simulation, cloud_center, m_use, cloudnum, case, snapshot, seed) 
        exit()

    try:
        data_bundle = Codedata(case, snapshot, m=1000, d=20, cloudnum=cloudnum, seed=seed)
        x_init = data_bundle.points[:m_use].copy()
    except NameError:
        print("Required external function (Codedata) is not defined. Cannot run integration.")
        exit()


    Pos = simulation.Pos - cloud_center
    VoronoiPos = simulation.VoronoiPos - cloud_center
    Pos, VoronoiPos = wrap_positions(Pos, VoronoiPos, simulation.Boxsize)

    # Forward
    print("Integrating forward...")
    pos_fwd = np.full((N_max, m_use, 3), np.nan)
    absb_fwd = np.full((N_max, m_use), np.nan)
    n_fwd = np.full((N_max, m_use), np.nan)

    current = x_init.copy()
    mask = np.ones(m_use, dtype=bool)
    b0, absB0, n0_mass, _ = find_points_and_get_fields(current, simulation.Bfield, simulation.Density, simulation.Density_grad, Pos, VoronoiPos)
    n0 = n0_mass * gr_cm3_to_nuclei_cm3

    pos_fwd[0] = current; absb_fwd[0] = absB0; n_fwd[0] = n0

    for step in range(1, N_max):
        if not np.any(mask):
            pos_fwd, absb_fwd, n_fwd = pos_fwd[:step], absb_fwd[:step], n_fwd[:step]
            break
        active_pos = current[mask]
        next_pos, _, _, _ = intmeth(active_pos, 0.01*np.ones(len(active_pos)), simulation.Bfield, simulation.Density, simulation.Density_grad, Pos, VoronoiPos, simulation.Volume)
        _, absB, n_mass, _ = find_points_and_get_fields(next_pos, simulation.Bfield, simulation.Density, simulation.Density_grad, Pos, VoronoiPos)
        n_new = n_mass * gr_cm3_to_nuclei_cm3
        pos_fwd[step, mask] = next_pos; absb_fwd[step, mask] = absB; n_fwd[step, mask] = n_new
        current[mask] = next_pos
        mask[mask] = (n_new >= density_threshold)

    # Backward
    print("Integrating backward...")
    pos_bck = np.full((N_max, m_use, 3), np.nan)
    absb_bck = np.full((N_max, m_use), np.nan)
    n_bck = np.full((N_max, m_use), np.nan)

    current = x_init.copy()
    mask = np.ones(m_use, dtype=bool)
    pos_bck[0] = current; absb_bck[0] = absB0; n_bck[0] = n0

    for step in range(1, N_max):
        if not np.any(mask):
            pos_bck, absb_bck, n_bck = pos_bck[:step], absb_bck[:step], n_bck[:step]
            break
        active_pos = current[mask]
        next_pos, _, _, _ = intmeth(active_pos, -0.01*np.ones(len(active_pos)), simulation.Bfield, simulation.Density, simulation.Density_grad, Pos, VoronoiPos, simulation.Volume)
        _, absB, n_mass, _ = find_points_and_get_fields(next_pos, simulation.Bfield, simulation.Density, simulation.Density_grad, Pos, VoronoiPos)
        n_new = n_mass * gr_cm3_to_nuclei_cm3
        pos_bck[step, mask] = next_pos; absb_bck[step, mask] = absB; n_bck[step, mask] = n_new
        current[mask] = next_pos
        mask[mask] = (n_new >= density_threshold)

    pos_full = np.concatenate([pos_bck[::-1], pos_fwd[1:]], axis=0)
    absb_full = np.concatenate([absb_bck[::-1], absb_fwd[1:]], axis=0)
    n_full = np.concatenate([n_bck[::-1], n_fwd[1:]], axis=0)

    # (Calculation remains the same: accumulates path length)
    dist_full = np.zeros_like(n_full)
    for i in range(m_use):
        for j in range(1, dist_full.shape[0]):
            if np.all(~np.isnan(pos_full[j, i])):
                step_pc = np.linalg.norm(pos_full[j, i] - pos_full[j-1, i])
                dist_full[j, i] = dist_full[j-1, i] + step_pc * pc_to_cm

    # Save 
    np.savez_compressed(
        out_path,
        seed_points=x_init,
        positions=pos_full,
        absB=absb_full,
        number_density=n_full, 
        distance_cm=dist_full
    )
    print(f"Saved to {out_path}")

    # Plot
    plot_func(dist_full, absb_full, pos_full, n_full, simulation, cloud_center, m_use, cloudnum, case, snapshot, seed)