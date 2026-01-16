import numpy as np
import os, time
from library import *

import pandas as pd


class Codedata:
    "will load the necessary information created in los_stats_forcord.py"

    def __init__(self, case, snapshot, m, d, cloudnum, seed):
        self.case     = case
        self.snapshot = snapshot
        self.m        = m
        self.d        = d
        self.cloudnum = cloudnum
        self.seed     = seed

        self.traj_los                    = None
        self.dens_los                    = None
        self.pos_los                     = None
        self.mean_column_densities_los   = None
        self.points                      = None
        self.crs_column                  = None
        self.los_column                  = None
        self.distance_los                = None
        self.number_density_los          = None
        self.number_density_crs          = None
        self.pos_crs                     = None
        self.local_fields_crs            = None
        self.abs_fields_crs              = None

        self._load_dataBundle()
        self._load_cloud_coords()

    def _load_dataBundle(self):
        'loads the data'
    
        databundlepath = os.path.join('thesis_los', self.case, self.snapshot)
        databundlename = f'DataBundle_MeanCD_andpathD_{self.seed}_{self.m}_{self.d}_{self.cloudnum}.npz'
        full_path_bundle = os.path.join(databundlepath, databundlename)

        print(f"Loading the data from: {full_path_bundle}")

        try:
            data = np.load(full_path_bundle, allow_pickle=True)
            self.traj_los                    = data["traj_los"]
            self.dens_los                    = data["dens_los"]
            self.pos_los                     = data["pos_los"]
            self.mean_column_densities_los   = data["mean_column_densities_los"]
            self.points                      = data["points"]
            self.crs_column                  = data["crs_column"]
            self.los_column                  = data["los_columns"]
            self.distance_los                = data["distance_LOS"]
            self.number_density_los          = data["number_density_LOS"]
            self.number_density_crs          = data["number_density_crs"]
            self.pos_crs                     = data["pos_crs"]
            self.local_fields_crs            = data["local_fields_crs"]
            self.abs_fields_crs              = data["absb_local_fields_crs"]

            print("data bundle loaded!")

        except FileNotFoundError:
            print("file not found")
            raise
    
    def _load_cloud_coords(self):
        "loads the cloud coordinates and stores core data."
        
        cloud_direc = os.path.join('clouds')
        coordinates_file = f'{self.case}_clouds.txt'
        full_cloud_path = os.path.join(cloud_direc, coordinates_file)
        
        try:
            df = pd.read_csv(full_cloud_path)
            cloud_index = int(self.cloudnum) 
            cloud_data = df.iloc[cloud_index]

            self.peak_density = cloud_data['Peak_Density']
            self.xcore = cloud_data['CloudCord_X']
            self.ycore = cloud_data['CloudCord_Y']
            self.zcore = cloud_data['CloudCord_Z']
            
        except FileNotFoundError:
            print("file not found")
            raise

    







