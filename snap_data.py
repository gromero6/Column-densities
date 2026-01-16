import os, time, csv, glob, sys
import h5py
import pandas as pd
from library import *

class snap_data:
    #load the snapshot data

    def __init__(self, snap, case, cloudnum):
        self.snap = snap
        self.case = case
        self.FloatType = np.float64
        self.IntType = np.int32
        self.cloudnum = cloudnum
        
        self.data = None
        self.clouds_centers = None
        self.Boxsize = None

        self.VoronoiPos = None
        self.Pos = None
        self.Bfield = None
        self.Preassure = None
        self.Velocities = None
        self.Density = None
        self.Mass = None
        self.Bfield_grad = None
        self.Density_grad = None
        self.Volume = None
        self.xcore = None
        self.ycore = None
        self.zcore = None
        self.peak_density = None

        self._load_snap()

    def _load_snap(self):
        
        # #troubleshooting: raises an error if the snapshot is not found
        # file_list = glob.glob(f'snap_{self.case}.hdf5')
        # print(file_list)
        # filename = None

        # for f in file_list:
        #     if self.snap in f:
        #         filename = f
        # if filename == None:
        #     raise FileNotFoundError
    
        #path for the clouds cordinates
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
        
        filename = f'snap_{self.snap}.hdf5'
        self.data = h5py.File(filename, 'r')
        print("Top-level groups/datasets:")
        for key in self.data.keys():
            print(f"  - {key}")
        self.Boxsize = self.data['Header'].attrs['BoxSize']

        # Directly convert and cast to desired dtype into the arrays that the functions will use
        self.VoronoiPos = np.asarray(self.data['PartType0']['Coordinates'], dtype=self.FloatType)
        self.Pos = np.asarray(self.data['PartType0']['CenterOfMass'], dtype=self.FloatType)
        self.Bfield = np.asarray(self.data['PartType0']['MagneticField'], dtype=self.FloatType)
        self.Pressure = np.asarray(self.data['PartType0']['Pressure'], dtype=self.FloatType)
        self.Velocities = np.asarray(self.data['PartType0']['Velocities'], dtype=self.FloatType)
        self.Density = np.asarray(self.data['PartType0']['Density'], dtype=self.FloatType)
        self.Mass = np.asarray(self.data['PartType0']['Masses'], dtype=self.FloatType)
        self.Bfield_grad = np.zeros((len(self.Pos), 9))
        self.Density_grad = np.zeros((len(self.Density), 3))
        self.Volume   = self.Mass/self.Density

