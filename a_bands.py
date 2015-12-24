# a_bands.py

import a_models as models
import sys
from a_config import *
from numpy import ma

band_option = 'rrc_'
b = 'rrc_'
b = 'rtoa_' # changed 20151204 (suggested by Kevin)

def just_get_netcdf(var):
    Scene = models.NetCDFFileModel(data_directory, scene_id)
    return Scene.connect_to_nc(var)

def get_var_before_mask(var):
    Scene = models.NetCDFFileModel(data_directory, scene_id)
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

def get_coastal():
    return get_var(b+'443')

def get_coastal_rtoa():
    return get_var('rtoa_'+'443')

def get_blue():
    return get_var(b+'483')

def get_green():
    return get_var(b+'561')

def get_red():
    return get_var(b+'655')

def get_nir():
    return get_var(b+'865')

def get_swir():
    return get_var(b+'1609')

def get_swir2():
    return get_var(b+'2201')

def get_cirrus():
    return get_var('rtoa_1373')

def get_temp():
    return get_var('BT_B10')

def get_bqa():
    return get_var('bqa')

def get_lat():
    return get_var('lat')

def get_lon():
    return get_var('lon')

# print(Scene.get_variables_list())

def calc_ndsi():
    green = get_green()
    swir = get_swir()
    return (green - swir)/(green + swir)

def calc_ndvi():
    nir = get_nir()
    red = get_red()
    return (nir - red)/(nir + red)