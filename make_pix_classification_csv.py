# input points as a csv file with columns as classification and rows as points
import sys
import numpy as np
from netCDF4 import Dataset
from numpy import ma
import csv
import pandas as pd
from osgeo import gdal
from os import listdir
import matplotlib.pyplot as pyplot

"""
Takes a file as an argument and opens it as an array.
"""

# dirname = r'/home/nicholas/Documents/data/Fmask/'
# filename = sys.argv[1]

def read_hdr_file(filename):
    ds = gdal.Open(filename                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 , gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    fmask_array = band.ReadAsArray()
    return fmask_array

"""
scene_id, pix_y, pix_x, reflectance, tests, classification, rgb_value, manual classification.
"""

data_directory = sys.argv[1]
scene_id = sys.argv[2]
# INPUT_FILE = sys.argv[3]
ref_data_directory = '/storage/Nicholas/Data/references/'
INPUT_FILE = ref_data_directory+scene_id + '_pixels.csv'
CSV_FILE = '/storage/Nicholas/Data/pixel_classification/' + scene_id + '_pixels' +'.csv'

fmask_filename = '/storage/Nicholas/ForNick/'+scene_id[6:9]+'/'+scene_id+'/'+scene_id+'_MTLFmask'
fmask = read_hdr_file(fmask_filename)

# classification_filename = '/storage/Nicholas/Data/' + scene_id + '_pcp_tirs_coastal.nc'
print('/storage/Nicholas/Data/pcp_ambiguous/20151221/' + scene_id + '_final_mask.nc')
classification_filename = '/storage/Nicholas/Data/pcp_ambiguous/20151221/' + scene_id + '_final_mask.nc'


nc_classification = Dataset(classification_filename)
classification = nc_classification.variables['final_mask'][:]

def open_csv_file():
    pass

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

band_option = 'rrc_'
b = 'rrc_'
b = 'rtoa_' # changed 20151204 (suggested by Kevin)

def just_get_netcdf(var):
    Scene = NetCDFFileModel(data_directory, scene_id)
    return Scene.connect_to_nc(var)

def get_var_before_mask(var):
    Scene = NetCDFFileModel(data_directory, scene_id)
    return Scene.get_data(var)

def get_mask_bqa():
    mask = get_var_before_mask('bqa')
    return mask    

mask = get_mask_bqa()

def get_var(var, mask=mask):
    '''
    Get the data from the requested variable band.
    TODO:
    Choose according to lat and lon values.
    '''
    # mask = get_mask_bqa()
    result = get_var_before_mask(var)
    result = ma.masked_where(mask==1, result)
    return result

b = 'rtoa_'
rtoa_coastal = get_var(b+'443')
rtoa_blue = get_var(b+'483')
rtoa_green = get_var(b+'561')
rtoa_red = get_var(b+'655')
rtoa_nir = get_var(b+'865')
rtoa_swir1 = get_var(b+'1609')
rtoa_swir2 = get_var(b+'2201')

cirrus = get_var('rtoa_1373')
temp1 = get_var('BT_B10')
temp2 = get_var('BT_B11')

b = 'rrc_'
rrc_coastal = get_var(b+'443')
rrc_blue = get_var(b+'483')
rrc_green = get_var(b+'561')
rrc_red = get_var(b+'655')
rrc_nir = get_var(b+'865')
rrc_swir1 = get_var(b+'1609')
rrc_swir2 = get_var(b+'2201')

def get_bqa():
    return get_var('bqa')

def get_lat():
    return get_var('lat')

def get_lon():
    return get_var('lon')

def calc_ndsi():
    return (rtoa_green - rtoa_swir1)/(rtoa_green + rtoa_swir1)

def calc_ndvi():
    return (rtoa_nir - rtoa_red)/(rtoa_nir + rtoa_red)

ndsi = calc_ndsi()
ndvi = calc_ndvi()
bqa = get_bqa()

def create_composite(red, green, blue):
    from scipy.misc import bytescale
    from utils import calculate_percentile


    #from skimage import data, img_as_float
    from skimage import exposure
    img_dim = red.shape
    img = np.zeros((img_dim[0], img_dim[1], 3), dtype=np.float)
    img[:,:,0] = red
    img[:,:,1] = green
    img[:,:,2] = blue
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(0, p98))
    return bytescale(img_rescale)

img = create_composite(rrc_red, rrc_green, rrc_blue)

my_list_headers = ['scene_id',
            'y',
            'x',
            'rtoa_coastal',
            'rtoa_blue',
            'rtoa_green',
            'rtoa_red',
            'rtoa_nir',
            'rtoa_swir1',
            'rtoa_swir2',
            'rrc_coastal',
            'rrc_blue',
            'rrc_green',
            'rrc_red',
            'rrc_nir',
            'rrc_swir1',
            'rrc_swir2',
            'cirrus',
            'temp1',
            'temp2',
            'bqa',
            'cirrus_test',
            'swir_test',
            'btc_test',
            'ndsi',
            'ndsi_test',
            'ndvi',
            'ndvi_test',
            'basic_test',
            'whiteness',
            'whiteness_test',
            'hot',
            'hot_test',
            'nir_swir',
            'nir_swir_test',
            'coastal_test',
            'ndsi_test',
            'sand_vs_cloud',
            'sand_vs_cloud_test',
            'rgb',
            'fmask',
            'classification',
            'manual_classification',
            ]

import os
if os.path.exists(CSV_FILE):
    print("File already exists")
else:
    with open(CSV_FILE, "w") as my_file:
        csvWriter = csv.writer(my_file, delimiter=',')
        csvWriter.writerow(my_list_headers)     

df = pd.read_csv(INPUT_FILE, header=0, sep=';')

def main_three():
    from scipy.misc import bytescale
    df = pd.read_csv(INPUT_FILE, header=0, sep=',')
    for index, row in df.iterrows():

        print(np.array(row[1][1:-1].split()))
        y,x = np.array(row[1][1:-1].split()).astype('int')
        if rtoa_coastal[y,x] != '--':
            manual_classification = row[2]
            # y,x = ast.literal_eval(row)
            # y = row[0]
            # x = row[1]
            cirrus_test = cirrus[y,x] > 0.01
            swir_test = rtoa_swir2[y,x] < 0.0215
            btc_test = temp1[y,x] < 300.15
            ndsi_test = ndsi[y,x] < 0.8
            ndvi_test = ndvi[y,x] < 0.8
            basic_test = np.logical_and.reduce((swir_test, btc_test, ndsi_test, ndvi_test))
            mean_vis = (rtoa_blue[y,x] + rtoa_green[y,x] + rtoa_red[y,x]) / 3
            whiteness = (np.abs((rtoa_blue[y,x] - mean_vis)/mean_vis) + 
                         np.abs((rtoa_green[y,x] - mean_vis)/mean_vis) +
                         np.abs((rtoa_red[y,x] - mean_vis)/mean_vis)
                         )
            whiteness_test = whiteness < 0.7
            hot = (1.0*rtoa_blue[y,x] - 0.5*rtoa_red[y,x]) - 0.08
            hot_test = hot > 0.0
            nir_swir = rtoa_swir1[y,x]/rtoa_nir[y,x]
            nir_swir_test = nir_swir > 0.75
            coastal_test = rtoa_coastal[y,x] > 0.2
            ndsi_test = ndsi[y,x] > -0.2
            sand_vs_cloud = (2*rtoa_coastal[y,x] - 3*rtoa_blue[y,x] + rtoa_green[y,x] ) / rtoa_blue[y,x]
            sand_vs_cloud_test = sand_vs_cloud < 0.2
            rgb = img[y, x, :]
            # fmask = fmask[y,x]
            # classification = column
            print('append list to csv file')
            my_list = [scene_id,
                        y,
                        x,
                        rtoa_coastal[y,x],
                        rtoa_blue[y,x],
                        rtoa_green[y,x],
                        rtoa_red[y,x],
                        rtoa_nir[y,x],
                        rtoa_swir1[y,x],
                        rtoa_swir2[y,x],
                        rrc_coastal[y,x],
                        rrc_blue[y,x],
                        rrc_green[y,x],
                        rrc_red[y,x],
                        rrc_nir[y,x],
                        rrc_swir1[y,x],
                        rrc_swir2[y,x],
                        cirrus[y,x],
                        temp1[y,x],
                        temp2[y,x],
                        bqa[y,x],
                        cirrus_test,
                        swir_test,
                        btc_test,
                        ndsi[y,x],
                        ndsi_test,
                        ndvi[y,x],
                        ndvi_test,
                        basic_test,
                        whiteness,
                        whiteness_test,
                        hot,
                        hot_test,
                        nir_swir,
                        nir_swir_test,
                        coastal_test,
                        ndsi_test,
                        sand_vs_cloud,
                        sand_vs_cloud_test,
                        rgb,
                        fmask[y,x],
                        classification[y,x],
                        manual_classification,
                        ]
            import os
            if os.path.exists(CSV_FILE):
                with open(CSV_FILE, "a+") as my_file:
                    csvWriter = csv.writer(my_file, delimiter=',')
                    csvWriter.writerow(my_list)
            else:
                with open(CSV_FILE, "w") as my_file:
                    csvWriter = csv.writer(my_file, delimiter=',')
                    csvWriter.writerow(my_list)
        else:
            print("Masked value in manual classification")

def main_two():
    df = pd.read_csv(INPUT_FILE, header=0, sep=',')
    for index, row in df.iterrows():
        print(np.array(row[1][1:-1].split()))
        y,x = np.array(row[1][1:-1].split()).astype('int')
        manual_classification = row[2]
        # y,x = ast.literal_eval(row)
        # y = row[0]
        # x = row[1]
        cirrus_test = cirrus[y,x] > 0.01
        swir_test = rtoa_swir2[y,x] < 0.0215
        btc_test = temp1[y,x] < 300.15
        ndsi_test = ndsi[y,x] < 0.8
        ndvi_test = ndvi[y,x] < 0.8
        basic_test = np.logical_and.reduce((swir_test, btc_test, ndsi_test, ndvi_test))
        mean_vis = (rtoa_blue[y,x] + rtoa_green[y,x] + rtoa_red[y,x]) / 3
        whiteness = (np.abs((rtoa_blue[y,x] - mean_vis)/mean_vis) + 
                     np.abs((rtoa_green[y,x] - mean_vis)/mean_vis) +
                     np.abs((rtoa_red[y,x] - mean_vis)/mean_vis)
                     )
        whiteness_test = whiteness < 0.7
        hot = (1.0*rtoa_blue[y,x] - 0.5*rtoa_red[y,x]) - 0.08
        hot_test = hot > 0.0
        nir_swir = rtoa_swir1[y,x]/rtoa_nir[y,x]
        nir_swir_test = nir_swir > 0.75
        coastal_test = rtoa_coastal[y,x] > 0.2
        ndsi_test = ndsi[y,x] > -0.2
        sand_vs_cloud = (2*rtoa_coastal[y,x] - 3*rtoa_blue[y,x] + rtoa_green[y,x] ) / rtoa_blue[y,x]
        sand_vs_cloud_test = sand_vs_cloud < 0.2
        # fmask = fmask[y,x]
        # classification = column
        print('append list to csv file')
        my_list = [scene_id,
                    y,
                    x,
                    rtoa_coastal[y,x],
                    rtoa_blue[y,x],
                    rtoa_green[y,x],
                    rtoa_red[y,x],
                    rtoa_nir[y,x],
                    rtoa_swir1[y,x],
                    rtoa_swir2[y,x],
                    rrc_coastal[y,x],
                    rrc_blue[y,x],
                    rrc_green[y,x],
                    rrc_red[y,x],
                    rrc_nir[y,x],
                    rrc_swir1[y,x],
                    rrc_swir2[y,x],
                    cirrus[y,x],
                    temp1[y,x],
                    temp2[y,x],
                    bqa[y,x],
                    cirrus_test,
                    swir_test,
                    btc_test,
                    ndsi_test,
                    ndvi_test,
                    basic_test,
                    whiteness,
                    whiteness_test,
                    hot,
                    hot_test,
                    nir_swir,
                    nir_swir_test,
                    coastal_test,
                    ndsi_test,
                    sand_vs_cloud,
                    sand_vs_cloud_test,
                    fmask[y,x],
                    classification[y,x],
                    manual_classification,
                    ]
        import os
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, "a+") as my_file:
                csvWriter = csv.writer(my_file, delimiter=',')
                csvWriter.writerow(my_list)
        else:
            with open(CSV_FILE, "w") as my_file:
                csvWriter = csv.writer(my_file, delimiter=',')
                csvWriter.writerow(my_list)           

def main():
    import ast
    for column in df:
        manual_classification = column
        for row in df[column]:
            y,x = ast.literal_eval(row)
            # y = row[0]
            # x = row[1]
            cirrus_test = cirrus[y,x] > 0.01
            swir_test = rtoa_swir2[y,x] < 0.0215
            btc_test = temp1[y,x] < 300.15
            ndsi_test = ndsi[y,x] < 0.8
            ndvi_test = ndvi[y,x] < 0.8
            basic_test = np.logical_and.reduce((swir_test, btc_test, ndsi_test, ndvi_test))
            mean_vis = (rtoa_blue[y,x] + rtoa_green[y,x] + rtoa_red[y,x]) / 3
            whiteness = (np.abs((rtoa_blue[y,x] - mean_vis)/mean_vis) + 
                         np.abs((rtoa_green[y,x] - mean_vis)/mean_vis) +
                         np.abs((rtoa_red[y,x] - mean_vis)/mean_vis)
                         )
            whiteness_test = whiteness < 0.7
            hot = (1.0*rtoa_blue[y,x] - 0.5*rtoa_red[y,x]) - 0.08
            hot_test = hot > 0.0
            nir_swir = rtoa_swir1[y,x]/rtoa_nir[y,x]
            nir_swir_test = nir_swir > 0.75
            coastal_test = rtoa_coastal[y,x] > 0.2
            ndsi_test = ndsi[y,x] > -0.2
            sand_vs_cloud = (2*rtoa_coastal[y,x] - 3*rtoa_blue[y,x] + rtoa_green[y,x] ) / rtoa_blue[y,x]
            sand_vs_cloud_test = sand_vs_cloud < 0.2
            # fmask = fmask[y,x]
            # classification = column
            print('append list to csv file')
            my_list = [scene_id,
                        y,
                        x,
                        rtoa_coastal[y,x],
                        rtoa_blue[y,x],
                        rtoa_green[y,x],
                        rtoa_red[y,x],
                        rtoa_nir[y,x],
                        rtoa_swir1[y,x],
                        rtoa_swir2[y,x],
                        rrc_coastal[y,x],
                        rrc_blue[y,x],
                        rrc_green[y,x],
                        rrc_red[y,x],
                        rrc_nir[y,x],
                        rrc_swir1[y,x],
                        rrc_swir2[y,x],
                        cirrus[y,x],
                        temp1[y,x],
                        temp2[y,x],
                        bqa[y,x],
                        cirrus_test,
                        swir_test,
                        btc_test,
                        ndsi_test,
                        ndvi_test,
                        basic_test,
                        whiteness,
                        whiteness_test,
                        hot,
                        hot_test,
                        nir_swir,
                        nir_swir_test,
                        coastal_test,
                        ndsi_test,
                        sand_vs_cloud,
                        sand_vs_cloud_test,
                        fmask[y,x],
                        classification[y,x],
                        manual_classification,
                        ]
            import os
            if os.path.exists(CSV_FILE):
                with open(CSV_FILE, "a+") as my_file:
                    csvWriter = csv.writer(my_file, delimiter=',')
                    csvWriter.writerow(my_list)
            else:
                with open(CSV_FILE, "w") as my_file:
                    csvWriter = csv.writer(my_file, delimiter=',')
                    csvWriter.writerow(my_list)                    

if __name__ == "__main__":
    main_three()