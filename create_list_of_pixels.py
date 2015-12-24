# create_list_of_pixels.py

from netCDF4 import Dataset
from scipy import misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import sys
import os

scene_id = sys.argv[1]
# CSV_FILE = sys.argv[2]
data_directory = '/storage/Nicholas/Data/references/'
CSV_FILE = data_directory+scene_id + '_pixels.csv'

data_directory = '/storage/Nicholas/Data/references/'

water_ref = color.rgb2gray(mpimg.imread(data_directory+scene_id+'_water.png'));
water_inds = np.where(water_ref<1)
wi = np.array(water_inds).T

cloud_ref = color.rgb2gray(mpimg.imread(data_directory+scene_id+'_clouds.png'));
cloud_inds = np.where(cloud_ref<1)
ci = np.array(cloud_inds).T

land_ref = color.rgb2gray(mpimg.imread(data_directory+scene_id+'_land.png'));
land_inds = np.where(land_ref<1)
li = np.array(land_inds).T

shadow_ref = color.rgb2gray(mpimg.imread(data_directory+scene_id+'_shadow.png'));
shadow_inds = np.where(shadow_ref<1)
si = np.array(shadow_inds).T

bright_ref = color.rgb2gray(mpimg.imread(data_directory+scene_id+'_sand.png'));
bright_inds = np.where(bright_ref<1)
bi = np.array(bright_inds).T
# How to read points
# np.array(example_point[1:-1].split(' ')).astype('int')


import csv
my_list_headers = ['scene_id',
                   'pixel',
                   'classification']

if os.path.exists(CSV_FILE):
    print("File already exists")
else:
    with open(CSV_FILE, "w") as my_file:
        csvWriter = csv.writer(my_file, delimiter=',')
        csvWriter.writerow(my_list_headers) 

for point in wi:
    line = [scene_id, point, 'water']
    with open(CSV_FILE, "a+") as my_file:
        csvWriter = csv.writer(my_file, delimiter=',')
        csvWriter.writerow(line)

for point in li:
    line = [scene_id, point, 'land']
    with open(CSV_FILE, "a+") as my_file:
        csvWriter = csv.writer(my_file, delimiter=',')
        csvWriter.writerow(line)

for point in ci:
    line = [scene_id, point, 'cloud']
    with open(CSV_FILE, "a+") as my_file:
        csvWriter = csv.writer(my_file, delimiter=',')
        csvWriter.writerow(line)

for point in si:
    line = [scene_id, point, 'shadow']
    with open(CSV_FILE, "a+") as my_file:
        csvWriter = csv.writer(my_file, delimiter=',')
        csvWriter.writerow(line)

for point in bi:
    line = [scene_id, point, 'sand']
    with open(CSV_FILE, "a+") as my_file:
        csvWriter = csv.writer(my_file, delimiter=',')
        csvWriter.writerow(line)