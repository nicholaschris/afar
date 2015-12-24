import operator
import pandas as pd
import numpy as np
from numpy import ma
from scipy.misc import imresize
import scipy.ndimage as ndimage
from skimage.morphology import disk, dilation

from a_bands import just_get_netcdf, storage_directory

def get_truth(input_one, input_two, comparison): # too much abstraction
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq}
    return ops[comparison](input_one, input_two)
    
def convert_to_celsius(brightness_temp_input):
    return brightness_temp_input - 272.15
    
def calculate_percentile(input_masked_array, percentile): 
    flat_fill_input = input_masked_array.filled(np.nan).flatten()
    df = pd.DataFrame(flat_fill_input)
    percentile = df.quantile(percentile/100.0)
    return percentile[0]
     
def save_object(obj, filename):
    import pickle
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)

def read_pkl(filename):
    # import pickle
    with open(filename, 'rb') as in_file:
        new_array = np.load(in_file)
    return new_array

def downsample(input_array, factor=4):
    output_array = input_array[::2, ::2] / 4 + input_array[1::2, ::2] / 4 + input_array[::2, 1::2] / 4 + input_array[1::2, 1::2] / 4
    return output_array

def dilate_boolean_array(input_array, disk_size=3):
    selem = disk(disk_size)
    dilated = dilation(input_array, selem)
    return dilated

def get_resized_array(img, size):
    lena = imresize(img, (size, size))
    return lena

def interp_and_resize(array, new_length):
    orig_y_length, orig_x_length = array.shape

    interp_factor_y = new_length / orig_y_length
    interp_factor_x = new_length / orig_x_length


    y = round(interp_factor_y * orig_y_length)
    x = round(interp_factor_x * orig_x_length)
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.mgrid.html
    new_indicies = np.mgrid[0:orig_y_length:y * 1j, 0:orig_x_length:x * 1j]
    # order=1 indicates bilinear interpolation.
    interp_array = ndimage.map_coordinates(array, new_indicies, 
                                           order=1, output=array.dtype)
    interp_array = interp_array.reshape((y, x))
    return interp_array

def parse_mtl(in_file):
    awesome = True
    f = open(in_file, 'r')
    print(in_file)
    mtl_dict = {}
    with open(in_file, 'r') as f:
        while awesome:
            line = f.readline()
            if line.strip() == '' or line.strip() == 'END':
                return mtl_dict
            elif 'END_GROUP' in line:
                pass
            elif 'GROUP' in line:
                curr_group = line.split('=')[1].strip()
                mtl_dict[curr_group] = {}
            else:
                attr, value = line.split('=')[0].strip(), line.split('=')[1].strip()
                mtl_dict[curr_group][attr] = value

def make_the_netcdf(outfile_name, input_data, data_directory, scene_id, description="", typeofarray=np.int8):
    from netCDF4 import Dataset
    full_path = data_directory # imported from bands from models
    # scene_id = scene_id # imported from bands from models from config etc...
    morus_hack = storage_directory
    # full_path = storage_directory # imported from bands from models
    print(full_path + scene_id+'_'+outfile_name+'.nc')
    outfile = Dataset(full_path + scene_id+'_'+outfile_name+'.nc', 'w', format='NETCDF4_CLASSIC')
    print(outfile.file_format)
    nc = just_get_netcdf('rrc_443')
    nclat = just_get_netcdf('lat')
    nclon = just_get_netcdf('lon')
    lat = nclat.variables['lat'][:]
    nclat.close()
    lon = nclon.variables['lon'][:]
    nclon.close()

    for item in nc.dimensions:
        outfile.createDimension(item, len(nc.dimensions[item]))

    outfile.createVariable(outfile_name, typeofarray, ('y','x'))
    outfile.variables[outfile_name][:] = input_data

    outfile.createVariable('lat', np.float32, ('y','x'))
    outfile.variables['lat'][:] = lat

    outfile.createVariable('lon', np.float32, ('y','x'))
    outfile.variables['lon'][:] = lon
    outfile.description = description

    for name in nc.ncattrs():
        # print(name)
        setattr(outfile, name, getattr(nc, name))
    print("File written to: " + full_path + scene_id+'_'+outfile_name+'.nc')
    nc.close()
    outfile.close()
