"""
Need a model that gets the SCENEID as an input
and then assumes all relevant files
for that SCENEID are in that folder
and does things that way.
"""

import sys
import numpy as np
from netCDF4 import Dataset

class NetCDFFileModel(object):

    def __init__(self, data_directory, scene_id):
        self.data_directory = data_directory
        self.scene_id = scene_id

    def connect_to_nc(self, var):
        full_path = self.data_directory + self.scene_id + '_' + var + '.nc'
        print(full_path)
        self.nc = Dataset(full_path, 'r')
        try:
            self.dimensions = self.nc.dimensions
            self.theta_v = self.nc.THV
            self.theta_0 = self.nc.TH0
            self.phi_v = self.nc.PHIV
            self.phi_0 = self.nc.PHI0
        except AttributeError as e:
            print("Couldn't get attributes")
        return self.nc

    def get_metadata(self, var):
        self.connect_to_nc(var)
        dimensions = self.dimensions = self.nc.dimensions
        theta_v = self.theta_v
        theta_0 = self.theta_0
        phi_v = self.phi_v
        phi_0 = self.phi_0
        nc.close()
        return dimensions, theta_v, theta_0, phi_v, phi_0

    def get_data(self, var):
        self.connect_to_nc(var)
        nc = self.nc
        result = np.array(nc.variables[var]).astype(np.float32)
        # result = np.array(nc.variables[var])
        nc.close()
        return result


    