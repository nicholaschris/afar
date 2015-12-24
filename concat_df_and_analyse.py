# plots_from_the_df.py
import numpy as np
import pandas as pd
import os
import sys
import matplotlib
from scipy import stats
# matplotlib.use('Agg')

import matplotlib.pyplot as plt 

INPUT_FILE1 = '/storage/Nicholas/Data/pixel_classification/LC81960302014022LGN00_pixels.csv'
INPUT_FILE2 = '/storage/Nicholas/Data/pixel_classification/LC81970222013186LGN00_pixels.csv'
INPUT_FILE3 = '/storage/Nicholas/Data/pixel_classification/LC81990242013280LGN00_pixels.csv'
INPUT_FILE4 = '/storage/Nicholas/Data/pixel_classification/LC82040212013251LGN00_pixels.csv'
header_ = 0

float_cols = ['rtoa_coastal',
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
            'ndsi_test',
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
            'fmask',
            'classification',
            ]

float32_cols = {c: np.float32 for c in float_cols}
df1 = pd.read_csv(INPUT_FILE1, engine='c', dtype=float32_cols, header = header_)
df2 = pd.read_csv(INPUT_FILE2, engine='c', dtype=float32_cols, header = header_) 
df3 = pd.read_csv(INPUT_FILE3, engine='c', dtype=float32_cols, header = header_) 
df4 = pd.read_csv(INPUT_FILE4, engine='c', dtype=float32_cols, header = header_) 

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
            'ndsi_test',
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

if header_ == None:            
    df.columns = my_list_headers

frames = [df1, df2, df3, df4]

df = pd.concat(frames)

water_classification = df[df.classification == 1]

scene_id = 'all_scenes'

def plot_a_thing_toa_errorbars_manual(_thing, ymax):
      """
      thing is a string
      http://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
      """
      thing = df[df['manual_classification']==str(_thing)]
      rtoa_thing = thing.ix[:,'rtoa_coastal':'rtoa_swir2']
      # rtoa_clouds_real = rtoa_clouds[rtoa_clouds.applymap(np.isreal).all(1)]
      rtoa_thing_real = rtoa_thing
      rtoa_thing_real = rtoa_thing_real.convert_objects(convert_numeric=True)
      # rtoa_thing_real = rtoa_thing_real[np.abs(rtoa_thing_real.Data-rtoa_thing_real.Data.mean())<=(3*rtoa_thing_real.Data.std())]
      # rtoa_thing_real[(np.abs(stats.zscore(rtoa_thing_real)) < 3).all(axis=1)]
      rtoa_thing_mean = rtoa_thing_real.mean()
      rtoa_thing_max = rtoa_thing_real.max()
      rtoa_thing_min = rtoa_thing_real.min()
      rtoa_thing_std = rtoa_thing_real.std()

      plt.close('all')
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, rtoa_thing_std, fmt='ok', lw=3)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, [rtoa_thing_mean - rtoa_thing_min, rtoa_thing_max - rtoa_thing_mean],
            fmt='.k', ecolor='gray', lw=1)
      labels = list(rtoa_thing_mean.keys())
      plt.xticks(list(range(len(rtoa_thing_max))), list(rtoa_thing_mean.keys()))
      # grow the y axis down by 0.05
      # ax.set_ylim(1.35, 1.8)
      # expand the x axis by 0.5 at two ends
      ax.set_xlim(-0.5, len(labels)-0.5)
      ax.set_ylim(0, ymax)
      ax.set_xticklabels( labels, rotation=45 )
      plt.title(scene_id + ' - Supervised classification - ' + _thing)      
      #plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
      fig.tight_layout()
      plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_'+_thing+ '_toa_errorbars_manual'+'.png')
      # Removing outliers


def plot_a_thing_toa_errorbars_fmask(_thing, ymax):
      """
      thing is a int
      http://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
      """
      classification = { 0.0 : 'land',
                         1.0 : 'water',
                         2.0 : 'shadow',
                         3.0 : 'snow',
                         4.0 : 'cloud',
                         255.0 : 'no_obs',

      }
      thing = df[df['fmask']==_thing]
      rtoa_thing = thing.ix[:,'rtoa_coastal':'rtoa_swir2']
      # rtoa_clouds_real = rtoa_clouds[rtoa_clouds.applymap(np.isreal).all(1)]
      rtoa_thing_real = rtoa_thing
      rtoa_thing_real = rtoa_thing_real.convert_objects(convert_numeric=True)
      # rtoa_thing_real = rtoa_thing_real[np.abs(rtoa_thing_real.Data-rtoa_thing_real.Data.mean())<=(3*rtoa_thing_real.Data.std())]
      # rtoa_thing_real[(np.abs(stats.zscore(rtoa_thing_real)) < 3).all(axis=1)]
      rtoa_thing_mean = rtoa_thing_real.mean()
      rtoa_thing_max = rtoa_thing_real.max()
      rtoa_thing_min = rtoa_thing_real.min()
      rtoa_thing_std = rtoa_thing_real.std()

      plt.close('all')
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, rtoa_thing_std, fmt='ok', lw=3)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, [rtoa_thing_mean - rtoa_thing_min, rtoa_thing_max - rtoa_thing_mean],
            fmt='.k', ecolor='gray', lw=1)
      labels = list(rtoa_thing_mean.keys())
      plt.xticks(list(range(len(rtoa_thing_max))), list(rtoa_thing_mean.keys()))
      # grow the y axis down by 0.05
      # ax.set_ylim(1.35, 1.8)
      # expand the x axis by 0.5 at two ends
      ax.set_xlim(-0.5, len(labels)-0.5)
      ax.set_ylim(0, ymax)
      ax.set_xticklabels( labels, rotation=45 )
      plt.title(scene_id + ' - Fmask classification - ' + classification[_thing])
      #plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
      fig.tight_layout()
      plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_'+classification[_thing]+ '_toa_errorbars_fmask'+'.png')



def plot_a_thing_toa_errorbars_classification(_thing, ymax):
      """
      thing is a int
      http://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
      """
      classification = { 0.0 : 'NaN',
                         1.0 : 'water',
                         2.0 : 'land',
                         3.0 : 'snow',
                         4.0 : 'shadow',
                         5.0 : 'cloud',
                         6.0 : 'cirrus',

      }
      thing = df[df['classification']==_thing]
      rtoa_thing = thing.ix[:,'rtoa_coastal':'rtoa_swir2']
      # rtoa_clouds_real = rtoa_clouds[rtoa_clouds.applymap(np.isreal).all(1)]
      rtoa_thing_real = rtoa_thing
      rtoa_thing_real = rtoa_thing_real.convert_objects(convert_numeric=True)
      # rtoa_thing_real = rtoa_thing_real[np.abs(rtoa_thing_real.Data-rtoa_thing_real.Data.mean())<=(3*rtoa_thing_real.Data.std())]
      # rtoa_thing_real[(np.abs(stats.zscore(rtoa_thing_real)) < 3).all(axis=1)]
      rtoa_thing_mean = rtoa_thing_real.mean()
      rtoa_thing_max = rtoa_thing_real.max()
      rtoa_thing_min = rtoa_thing_real.min()
      rtoa_thing_std = rtoa_thing_real.std()

      plt.close('all')
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, rtoa_thing_std, fmt='ok', lw=3)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, [rtoa_thing_mean - rtoa_thing_min, rtoa_thing_max - rtoa_thing_mean],
            fmt='.k', ecolor='gray', lw=1)
      labels = list(rtoa_thing_mean.keys())
      plt.xticks(list(range(len(rtoa_thing_max))), list(rtoa_thing_mean.keys()))
      # grow the y axis down by 0.05
      # ax.set_ylim(1.35, 1.8)
      # expand the x axis by 0.5 at two ends
      ax.set_ylim(0, ymax)
      ax.set_xlim(-0.5, len(labels)-0.5)
      ax.set_xticklabels( labels, rotation=45 )
      plt.title(scene_id + ' - Unsupervised classification - ' + classification[_thing])
      #plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
      fig.tight_layout()
      plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_'+classification[_thing]+ '_toa_errorbars_classification'+'.png')

def plot_a_thing_toa_errorbars_classification_two(_thing, _thing2, ymax):
      """
      thing is a int
      http://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
      """
      classification = { 0.0 : 'NaN',
                         1.0 : 'water',
                         2.0 : 'land',
                         3.0 : 'snow',
                         4.0 : 'shadow',
                         5.0 : 'cloud',
                         6.0 : 'cirrus',

      }
      thing = df[(df['classification']==_thing) | (df['classification']==_thing2)]
      rtoa_thing = thing.ix[:,'rtoa_coastal':'rtoa_swir2']
      # rtoa_clouds_real = rtoa_clouds[rtoa_clouds.applymap(np.isreal).all(1)]
      rtoa_thing_real = rtoa_thing
      rtoa_thing_real = rtoa_thing_real.convert_objects(convert_numeric=True)
      # rtoa_thing_real = rtoa_thing_real[np.abs(rtoa_thing_real.Data-rtoa_thing_real.Data.mean())<=(3*rtoa_thing_real.Data.std())]
      # rtoa_thing_real[(np.abs(stats.zscore(rtoa_thing_real)) < 3).all(axis=1)]
      rtoa_thing_mean = rtoa_thing_real.mean()
      rtoa_thing_max = rtoa_thing_real.max()
      rtoa_thing_min = rtoa_thing_real.min()
      rtoa_thing_std = rtoa_thing_real.std()

      plt.close('all')
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, rtoa_thing_std, fmt='ok', lw=3)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, [rtoa_thing_mean - rtoa_thing_min, rtoa_thing_max - rtoa_thing_mean],
            fmt='.k', ecolor='gray', lw=1)
      labels = list(rtoa_thing_mean.keys())
      plt.xticks(list(range(len(rtoa_thing_max))), list(rtoa_thing_mean.keys()))
      # grow the y axis down by 0.05
      # ax.set_ylim(1.35, 1.8)
      # expand the x axis by 0.5 at two ends
      ax.set_ylim(0, ymax)
      ax.set_xlim(-0.5, len(labels)-0.5)
      ax.set_xticklabels( labels, rotation=45 )
      plt.title(scene_id + ' - Unsupervised classification - ' + classification[_thing]+'_'+classification[_thing2])
      #plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
      fig.tight_layout()
      plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_'+classification[_thing]+'_'+classification[_thing2]+ '_toa_errorbars_classification'+'.png')



def plot_scatter(x,y):
      plt.scatter(x, y, s=1, marker=",",linewidth='0') #,facecolors=z)
      ax = plt.gca()
      ax.set_axis_bgcolor('black')
      # plt.xlim(-0.1, 1.1)
      # plt.ylim(-0.1, 1.1)
      # plt.ylabel('RC 443', fontsize=16)
      # plt.xlabel('RC 1609', fontsize=16)
      plt.axis("tight")
# plt.savefig(data_directory+'coastal_threshold/'+scene_id+'_443_1609_scatter.png', dpi=300)
# plt.close('all')

# rgb_values = list(df.rgb.values)
# rgb_values = df.rgb.values
# for index, item in enumerate(rgb_values):
#     rgb_values[index] = [int(y) for y in (item.strip("'['").strip("']'").split())]

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

def plot_a_thing_toa_errorbars_manual(_thing, ymax):
      """
      thing is a string
      http://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
      """
      thing = df[df['manual_classification']==str(_thing)]
      rtoa_thing = thing.ix[:,'rtoa_coastal':'rtoa_swir2']
      # rtoa_clouds_real = rtoa_clouds[rtoa_clouds.applymap(np.isreal).all(1)]
      rtoa_thing_real = rtoa_thing
      rtoa_thing_real = rtoa_thing_real.convert_objects(convert_numeric=True)
      # rtoa_thing_real = rtoa_thing_real[np.abs(rtoa_thing_real.Data-rtoa_thing_real.Data.mean())<=(3*rtoa_thing_real.Data.std())]
      # rtoa_thing_real[(np.abs(stats.zscore(rtoa_thing_real)) < 3).all(axis=1)]
      rtoa_thing_mean = rtoa_thing_real.mean()
      rtoa_thing_max = rtoa_thing_real.max()
      rtoa_thing_min = rtoa_thing_real.min()
      rtoa_thing_std = rtoa_thing_real.std()

      plt.close('all')
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, rtoa_thing_std, fmt='ok', lw=3)
      ax.errorbar(list(range(len(rtoa_thing_max))), rtoa_thing_mean, [rtoa_thing_mean - rtoa_thing_min, rtoa_thing_max - rtoa_thing_mean],
            fmt='.k', ecolor='gray', lw=1)
      labels = list(rtoa_thing_mean.keys())
      labels = ['443', '483', '561', '655', '865', '1373', '1609', '2201']
      plt.xticks(list(range(len(rtoa_thing_max))), list(rtoa_thing_mean.keys()))
      # grow the y axis down by 0.05
      # ax.set_ylim(1.35, 1.8)
      # expand the x axis by 0.5 at two ends
      ax.set_xlim(-0.5, len(labels)-0.5)
      ax.set_ylim(0, ymax)
      ax.set_xticklabels( labels, rotation=45 )
      plt.xlabel('Wavelength (nm)')
      plt.ylabel('Top of Atmosphere Reflectance')
      plt.title('Supervised classification - ' + _thing)      
      #plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
      fig.tight_layout()
      plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_'+_thing+ '_toa_errorbars_manual'+'.png', dpi=300)
      # Removing outliers

def main_manual():
      # df.to_csv('/storage/Nicholas/Data/pixel_classification/pixels.csv')
      plot_a_thing_toa_errorbars_manual('cloud', 1.5)
      plot_a_thing_toa_errorbars_manual('land', 1.5)
      plot_a_thing_toa_errorbars_manual('water', 0.25)
      plot_a_thing_toa_errorbars_manual('shadow', 0.25)
      plot_a_thing_toa_errorbars_manual('sand', 1.5)

def main():
      # df.to_csv('/storage/Nicholas/Data/pixel_classification/pixels.csv')
      plot_a_thing_toa_errorbars_manual('cloud', 2)
      plot_a_thing_toa_errorbars_manual('land', 0.8)
      plot_a_thing_toa_errorbars_manual('water', 0.25)
      plot_a_thing_toa_errorbars_manual('shadow', 0.5)
      plot_a_thing_toa_errorbars_manual('sand', 2.0)
      plot_a_thing_toa_errorbars_fmask(0, 0.8)
      plot_a_thing_toa_errorbars_fmask(1, 0.25)
      plot_a_thing_toa_errorbars_fmask(2, 0.5)
      plot_a_thing_toa_errorbars_fmask(4, 2)
      plot_a_thing_toa_errorbars_classification(1, 0.25)
      plot_a_thing_toa_errorbars_classification(2, 0.8)
      plot_a_thing_toa_errorbars_classification(4, 0.5)
      plot_a_thing_toa_errorbars_classification(5, 2)
      plot_a_thing_toa_errorbars_classification(6, 1.5)
      plot_a_thing_toa_errorbars_classification_two(5, 6, 2)

# def print_means_and_std():

#       print(cloud.rtoa_swir1.mean())
#       print(cloud.rtoa_swir1.std())
#       print(cloud.ndsi.mean())
#       print(cloud.ndsi.std())
#       print(cloud.ndvi.mean())
#       print(cloud.ndvi.std())
#       print(cloud.whiteness.mean())
#       print(cloud.whiteness.std())
#       print(cloud.hot.mean())
#       print(cloud.hot.std())
#       print(cloud.nir_swir.mean())
#       print(cloud.nir_swir.std())
#       print(cloud.rtoa_coastal.mean())
#       print(cloud.rtoa_coastal.std())

#       print(water.rtoa_swir1.mean())
#       print(water.rtoa_swir1.std())
#       print(water.ndsi.mean())
#       print(water.ndsi.std())
#       print(water.ndvi.mean())
#       print(water.ndvi.std())
#       print(water.whiteness.mean())
#       print(water.whiteness.std())
#       print(water.hot.mean())
#       print(water.hot.std())
#       print(water.nir_swir.mean())
#       print(water.nir_swir.std())
#       print(water.rtoa_coastal.mean())
#       print(water.rtoa_coastal.std())

#       print(sand.rtoa_swir1.mean())
#       print(sand.rtoa_swir1.std())
#       print(sand.ndsi.mean())
#       print(sand.ndsi.std())
#       print(sand.ndvi.mean())
#       print(sand.ndvi.std())
#       print(sand.whiteness.mean())
#       print(sand.whiteness.std())
#       print(sand.hot.mean())
#       print(sand.hot.std())
#       print(sand.nir_swir.mean())
#       print(sand.nir_swir.std())
#       print(sand.rtoa_coastal.mean())
#       print(sand.rtoa_coastal.std())

#       print(land.rtoa_swir1.mean())
#       print(land.rtoa_swir1.std())
#       print(land.ndsi.mean())
#       print(land.ndsi.std())
#       print(land.ndvi.mean())
#       print(land.ndvi.std())
#       print(land.whiteness.mean())
#       print(land.whiteness.std())
#       print(land.hot.mean())
#       print(land.hot.std())
#       print(land.nir_swir.mean())
#       print(land.nir_swir.std())
#       print(land.rtoa_coastal.mean())
#       print(land.rtoa_coastal.std())

#       print(shadow.rtoa_swir1.mean())
#       print(shadow.rtoa_swir1.std())
#       print(shadow.ndsi.mean())
#       print(shadow.ndsi.std())
#       print(shadow.ndvi.mean())
#       print(shadow.ndvi.std())
#       print(shadow.whiteness.mean())
#       print(shadow.whiteness.std())
#       print(shadow.hot.mean())
#       print(shadow.hot.std())
#       print(shadow.nir_swir.mean())
#       print(shadow.nir_swir.std())
#       print(shadow.rtoa_coastal.mean())
#       print(shadow.rtoa_coastal.std())

