# afar.py

from a_bands import *
import a_utils as utils
import numpy as np
from numpy import ma
from netCDF4 import Dataset
from math import pi, tan, cos, sin
from skimage.morphology import reconstruction
from multiprocessing import Process
from datetime import datetime
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from some_kind_of_local_config_file_that_allows_the_thresholds_to_be_changed import *
startTime = datetime.now()

"""
This script performs an adapted pcp from Fmask, then removes some pcp pixels using caostal band and ndsi threshold and then projects clouds 
using pcp_strict and then creates cloud masks...
"""

cirrus = get_cirrus()
coastal = get_coastal()
swir = get_swir()
swir2 = get_swir2()
temp = get_temp()
blue = get_blue()
green = get_green()
red = get_red()
nir = get_nir()

def calc_ndsi():
    return (green - swir)/(green + swir)

def calc_ndvi():
    return (nir - red)/(nir + red)
    
ndvi = calc_ndvi()
ndsi = calc_ndsi()

# temp in Kelvin
# Threshold taken from FMASK improved (2015)
snow = np.logical_and.reduce(((ndsi > 0.4), (red > 0.11), (blue > 0.1), (temp<283)))

def cirrus_test():
    # cirrus = get_cirrus()
    return cirrus>0.01

def calc_cirrus_prob(cirrus=cirrus):
    """
    Cirrus probability is used for new PCL in Zhu and Woodcock 2015
    """
    return cirrus / 0.04
    
def swir_test(swir=swir2):
    """
    From Quinten.
    Used in Acolite?
    """
    return swir > 0.0215

def cirrus_test(cirrus=cirrus):
    """
    cirrus ~ 1373nm
    Returns the cirrus test from Zhu and Woodcock 2015.
    Useful if there is a cirrus band and not TIRS bands.
    e.g. Sentinel 2
    """
    return cirrus>0.01

def convert_to_celsius(brightness_temp_input):
    return brightness_temp_input - 272.15

def calc_basic_test_with_temp_and_cirrus(swir2=swir2, temp=temp, ndsi=ndsi, ndvi=ndvi):
    """
    The basic test from Zhu and Woodcock 2014.
    Needs TIRS bands.
    Band 7 is the same for ETM+ and OLI.
    Seems to underestimate? So change swir to 0.0215
    """
    cirrus_test = cirrus>0.01
    band_7_test = swir2 > 0.0215
    btc_test = convert_to_celsius(temp) < 27.0
    ndsi_test = ndsi < 0.8
    ndvi_test = ndvi < 0.8 
    basic_test_wo_cirrus = np.logical_and.reduce((band_7_test,
                                        ndsi_test,
                                        ndvi_test))
    basic_test = np.logical_or(basic_test_wo_cirrus, cirrus_test)
    return basic_test

def calc_whiteness(blue=blue, green=green, red=red):
    """
    From Zhu and Woodcock 2014.
    Whiteness test seems to include a lot of pixels. Might be safe to exclude.
    0.7 is the optimal threshold for Landsat 7 but haven't tried L8.
    Whiteness does not identify cloud shadow pixels!
    Turbid water is very "white".
    0.75, is this too high? 
    Atmospheric correction is approximately how much?
    Should we get these values from Acolite?
    Quinten suggested to add swir and nir to the test...
    """
    mean_vis = (blue + green + red) / 3
    whiteness = (np.abs((blue - mean_vis)/mean_vis) + 
                 np.abs((green - mean_vis)/mean_vis) +
                 np.abs((red - mean_vis)/mean_vis)
                 )
    # whiteness[np.where(whiteness>1)] = 1
    return whiteness

def calc_whiteness_test():
    """
    The whiteness test.
    It is supposed to exclude pixels not white enough to be clouds but it isn't working.
    Actually seems to work. Test out Singapore Scene tho...
    Change the threshold to 2 for adding swir and nir....
    """
    whiteness_test = calc_whiteness() < 0.7 # 0.7 in paper
    return whiteness_test

def calculate_hot(blue=blue, red=red):
    """
    Hot test results in omission of cloud pixels. 
    Trying with RTOA.
    Also sometimes produces funky results.
    Generally it catches the underestimated cloud pixels from basic test.
    Threshold should be tested.
    Make sure to use blue and red!
    Try again with rrc... Looks good... if remove 0.08 (atm correction?)
    What is the 0.08 from? RMSE between rrc and rtoa is approximately 0.08..
    How are these numbers ( a and b ) derived?
    So with TOA or RRC? 
    And if with RRC with or without - 0.08?
    """
    a = 1
    b = 0.5
    hot_test = (a*blue - b*red) - 0.08
    return hot_test

def calc_hot_test():
    """
    Make sure to use blue and red!
    """
    hot_test = calculate_hot() > 0.0
    return hot_test

def calc_swirnir(nir=nir, swir=swir):
    """
    b4/b5 in Zhu 2012
    swir/nir test seems to include a lot of pixels. Might be safe to exclude.
    Includes more pixels if 2201 is used in comparison to 1609.
    Zhu and Woodcock use Swir 1 but Swir 2 catches more pixles?
    They use threshold of 0.75. Here we try 0.7. And then we try 0.75.
    Supposed to exclude bright rock and desert.
    UPDATE: Swapping to use swir1 because VITO uses it.
    Is it pixels exceeding or pixels under?
    """
    return (nir/swir)

def calc_swir_nir_test():
    return calc_swirnir() > 0.75

basic_test_result = calc_basic_test_with_temp_and_cirrus()
hot_result = calc_hot_test()
whiteness_result = calc_whiteness_test()
swir_nir_result = calc_swir_nir_test()

def create_cm_blues():
    from matplotlib import cm
    import numpy as np
    theCM = cm.get_cmap('Blues')
    theCM._init() # this is a hack to get at the _lut array, which stores RGBA vals
    alphas = np.abs(np.linspace(0, .9, theCM.N))
    theCM._lut[:-3,-1] = alphas
    return theCM

def create_cm_greys():
    from matplotlib import cm
    import numpy as np
    theCM = cm.get_cmap('Greys')
    theCM._init() # this is a hack to get at the _lut array, which stores RGBA vals
    alphas = np.abs(np.linspace(0, .9, theCM.N))
    theCM._lut[:-3,-1] = alphas
    return theCM

def create_cm_oranges():
    from matplotlib import cm
    import numpy as np
    theCM = cm.get_cmap('Oranges')
    theCM._init() # this is a hack to get at the _lut array, which stores RGBA vals
    alphas = np.abs(np.linspace(0, .9, theCM.N))
    theCM._lut[:-3,-1] = alphas
    return theCM

def create_cm_greens():
    from matplotlib import cm
    import numpy as np
    theCM = cm.get_cmap('Greens')
    theCM._init() # this is a hack to get at the _lut array, which stores RGBA vals
    alphas = np.abs(np.linspace(0, .9, theCM.N))
    theCM._lut[:-3,-1] = alphas
    return theCM


def create_composite(red, green, blue):
    from scipy.misc import bytescale
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

img = create_composite(red, green, blue)
from skimage import exposure
img_rescale = exposure.rescale_intensity(img, in_range=(0, 95))

CMO = create_cm_oranges()
CMR = create_cm_greens()
CMB = create_cm_blues()
CMG = create_cm_greys()

def calc_pcp():
    return np.logical_and.reduce((basic_test_result, hot_result, whiteness_result, swir_nir_result))

cirrus_test = cirrus>0.01
pcp = calc_pcp()

# Thresholds estimated from manually classified spectra
pcp_strict = np.logical_and.reduce((pcp, coastal > 0.2, ndsi>-0.17))

def water_test_fmask(ndvi=ndvi, nir=nir, swir=swir):
    """
    From Zhu and Woodcock 2014.
    How well does it work in Extremely Turbid Waters?
    """
    water_condition_one = np.logical_and((ndvi < 0.01), (nir > 0.11))
    water_condition_two = np.logical_and((ndvi < 0.1), (nir < 0.05))
    water_test = np.logical_or(water_condition_one, water_condition_two)
    return water_test

def water_test_swir(swir=swir):
    swir_test = swir < 0.0215
    return swir_test

water = water_test_swir() # suggestion by Kevin
mask = get_mask_bqa()
water = ma.masked_where(mask==1, water)


def bresenham(origin, dest):
    # debug code
    print(origin)
    print(dest)
    # end debug code
    x0 = origin[0]; y0 = origin[1]
    x1 = dest[0]; y1 = dest[1]
    steep = abs(y1 - y0) > abs(x1 - x0)

    if steep: 
        x0, y0 = y0, x0 
        x1, y1 = y1, x1 

    backward = x0 > x1 
    if backward: 
        x0, x1 = x1, x0 
        y0, y1 = y1, y0 

    dx = x1 - x0
    dy = abs(y1 - y0)
    error = dx / 2
    y = y0

    if y0 < y1: 
      ystep = 1 
    else: 
      ystep = -1

    result = []
    print("x0 = %d" % (x0))
    print("x1 = %d" % (x1))
    print("y0 = %d" % (y0))
    print("y1 = %d" % (y1))
    for x in range(int(x0), int(x1)):
        if steep: result.append((y,x))
        else: result.append((x,y))
        error -= dy
        if error < 0:
            y += ystep
            error += dx 
    if backward: 
      result.reverse()
    print(result)
    return(result)
    
def get_the_netcdf(var_name, mask=mask):
    from netCDF4 import Dataset
    print("Getting NetCDF file from: " + data_directory+scene_id+'_'+var_name+'.nc')
    nc = Dataset(data_directory+scene_id+'_'+var_name+'.nc')
    out_array = nc.variables[var_name][:]
    out_array = ma.masked_where(mask==1, out_array)
    nc.close()
    return out_array.astype(np.int8)

def get_the_netcdf_metadata(var_name, mask=mask):
    from netCDF4 import Dataset
    morus_hack = '../../nicholas_data/'
    print(data_directory+scene_id+'_'+var_name+'.nc')
    nc = Dataset(data_directory+scene_id+'_'+var_name+'.nc')
    dimensions = nc.dimensions
    theta_v = nc.THV
    theta_0 = nc.TH0
    phi_v = nc.PHIV
    phi_0 = nc.PHI0
    nc.close()
    return dimensions, theta_v, theta_0, phi_v, phi_0

from skimage import morphology
dimensions, theta_v, theta_0, phi_v, phi_0 = get_the_netcdf_metadata('bqa')
opening_selem = 1
closing_selem = 3
o_selem   = morphology.disk(opening_selem)
pcp = morphology.opening(pcp, o_selem)
c_selem   = morphology.disk(closing_selem)
bin_im = morphology.closing(pcp, c_selem)

buffered_pcp = utils.dilate_boolean_array(bin_im)
buffered_pcp_strict = utils.dilate_boolean_array(pcp_strict)

th0 = theta_0
phi0 = pi - phi_0 # 180deg - azimuth angle

def max_x_y_offset(th0, phi0, max_height=5000):
    max_cloud_height = max_height
    d = max_cloud_height/30 # cloud_height(label_no)/30
    x_offset = - d*tan(th0)*sin(phi0)
    y_offset = - d*tan(th0)*cos(phi0)
    return x_offset, y_offset

x_offset, y_offset = max_x_y_offset(th0, phi0)
x_offset_cirrus, y_offset_cirrus = max_x_y_offset(th0, phi0, max_height=10000)

def find_maxmin(offset):
    if offset <=0:
        var_min = np.abs(np.int(offset))
        var_max=0
    else:
        var_min=0
        var_max = np.int(offset)
    return var_min, var_max

def shift_shifted(input_array, x_offset, y_offset):
    original_data = input_array
    original_shape = input_array.shape
    expanded_shape = input_array.shape[0] + abs(int(y_offset)), input_array.shape[1] + abs(int(x_offset))
    expanded=np.zeros(expanded_shape)
    ymin, ymax = find_maxmin(y_offset)
    xmin, xmax = find_maxmin(x_offset)
    print('Expanded shape is {0} and original data shape is {1} while the x and y offsets are {2} and {3}, with angles {4} and {5}'.format(expanded_shape, original_shape, x_offset, y_offset, th0, phi0))
    expanded[ymin:expanded_shape[0]-ymax, xmin:expanded_shape[1]-xmax] = original_data
    cloud_object_inds = np.where(expanded==1)
    origin = (0, 0)
    dest = (int(x_offset), int(y_offset))
    line = bresenham(origin, dest)
    for x, y in line:
        print('{0} and {1} out of {2} and {3}, with {4} and {5}'.format(x, y, x_offset, y_offset, th0, phi0))
        x_inds = cloud_object_inds[1] + x
        y_inds = cloud_object_inds[0] + y
        expanded[y_inds, x_inds] = 1
    crop = expanded[ymin:expanded_shape[0]-ymax, xmin:expanded_shape[1]-xmax]
    return crop



cirrus = None
coastal = None
swir = None
swir2 = None
# temp = get_temp()
ndvi = None
blue = None
green = None
red = None
nir = None
ndsi = None



pcp_shifted = shift_shifted(buffered_pcp_strict, x_offset, y_offset)
cirrus_shifted = shift_shifted(cirrus_test, x_offset_cirrus, y_offset_cirrus)

cloud_shadow = np.logical_or(pcp_shifted, cirrus_shifted)
description_dictionary = {
    'NaN':0,
    'Water':1,
    'Land':2,
    'Snow':3,
    'Cloud Shadow':4,
    'Cloud':5,
    'Cirrus':6
}
final_mask = np.zeros(pcp.shape)
final_mask[np.where(water==0)] = 2
final_mask[np.where(snow==1)] = 3
final_mask[np.where(water==1)] = 1
final_mask[np.where(buffered_pcp==1)] = 5
final_mask[np.where(cloud_shadow==1)] = 4
final_mask[np.where(buffered_pcp_strict==1)] = 6
final_mask[np.where(cirrus_test==1)] = 7
final_mask[np.where(mask==1)] = 0

from copy import copy
import matplotlib


my_norm = matplotlib.colors.Normalize(vmin=0.01, vmax=7, clip=False)

import matplotlib as mpl
_cmap = mpl.colors.ListedColormap(['#1abc9c', '#95a5a6', '#bdc3c7', '#2c3e50', '#e67e22', '#d35400', '#f1c40f'])
my_cmap = copy(plt.get_cmap(_cmap)) # make a copy so we don't mess up system copy
my_cmap.set_under('r', alpha=.5) # make locations over vmax translucent red
my_cmap.set_over('w', alpha=0)   # make location under vmin transparent white
my_cmap.set_bad('k')             # make location with invalid data black


import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def colorbar_index(ncolors):
    cmap = mpl.colors.ListedColormap(['#1abc9c', '#95a5a6', '#bdc3c7', '#2c3e50', '#e67e22', '#d35400', '#f1c40f'])
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(['Water', 'Land', 'Snow', 'Cloud Shadow', 'Possible Clouds', 'Cloud', 'Cirrus'])

def plot_help(data, cmap, title):
    data = ma.masked_where(data==0, data)
    import matplotlib as mpl
    bounds = [1, 2, 3, 4, 5, 6, 7, 8]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    cax = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    import matplotlib.cm as cm
    colorbar_index(ncolors=7)
    plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
    plt.savefig(storage_directory+scene_id+'_' + title + '.png', dpi=300)

def plot_help_no_frame(data, cmap, title):
    data = ma.masked_where(data==0, data)
    import matplotlib as mpl
    bounds = [1, 2, 3, 4, 5, 6, 7, 8]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(data, cmap=cmap, norm=norm)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig(storage_directory+scene_id+'_' + title + '.png', dpi=300, bbox_inches='tight', pad_inches=0)

def main():

    _description = """Landsat Pixel Identification and cloud mask.\n
    'NaN':0,\n
    'Water':1,\n
    'Land':2,\n
    'Snow':3,\n
    'Cloud Shadow':4,\n
    'Cloud':5,\n
    'Cirrus':6\n

    Done by N Pringle at RBINS
    """
    utils.make_the_netcdf('final_mask', final_mask, storage_directory, scene_id, description=_description, typeofarray=np.int8)
    plot_help(final_mask, my_cmap, b+'afar')
    plot_help_no_frame(final_mask, my_cmap, b+'afar_noframe')
    print("\n\nCompleted in: {0} at {1}".format(datetime.now() - startTime, datetime.now()))

def plot_test_results():
    plt.close('all')
    plt.imshow(img_rescale)
    plt.imshow(coastal > 0.2, cmap=CMO, alpha=0.8)
    title = 'coastal_result'
    plt.savefig(storage_directory+scene_id+'_' + title +'.png', dpi=300)

    plt.close('all')
    plt.imshow(img_rescale)
    plt.imshow(basic_test_result, cmap=CMO, alpha=0.8)
    title = 'basic_test_result'
    plt.savefig(storage_directory+scene_id+'_' + title +'.png', dpi=300)

    plt.close('all')
    plt.imshow(img_rescale)
    plt.imshow(hot_result, cmap=CMO, alpha=0.8)
    title = 'hot_result'
    plt.savefig(storage_directory+scene_id+'_' + title +'.png', dpi=300)

    plt.close('all')
    plt.imshow(img_rescale)
    plt.imshow(whiteness_result, cmap=CMO, alpha=0.8)
    title = 'whiteness_result'
    plt.savefig(storage_directory+scene_id+'_' + title +'.png', dpi=300)

    plt.close('all')
    plt.imshow(img_rescale)
    plt.imshow(calc_swirnir() > 0.75, cmap=CMO, alpha=0.8)
    title = 'swir_nir_result'
    plt.savefig(storage_directory+scene_id+'_' + title +'.png', dpi=300)

    plt.close('all')
    plt.imshow(img_rescale)
    plt.imshow(coastal > 0.2, cmap=CMO, alpha=0.8)
    title = 'coastal_test'
    plt.savefig(storage_directory+scene_id+'_' + title +'.png', dpi=300)

    plt.close('all')
    plt.imshow(img_rescale)
    plt.imshow(ndsi>-0.15, cmap=CMO, alpha=0.8)
    title = 'ndsi_test'
    plt.savefig(storage_directory+scene_id+'_' + title +'.png', dpi=300)

    plt.close('all')
    plt.imshow(img_rescale)
    plt.imshow(cirrus>0.01, cmap=CMO, alpha=0.8)
    title = 'cirrus_test'
    plt.savefig(storage_directory+scene_id+'_' + title +'.png', dpi=300)

    plt.close('all')
    plt.imshow(img_rescale)
    plt.imshow(water, cmap=CMO, alpha=0.8)
    title = 'water_test'
    plt.savefig(storage_directory+scene_id+'_' + title +'.png', dpi=300)

if __name__ == "__main__":
    main()
