# plots_from_the_df.py
import numpy as np
import pandas as pd
import os
import sys
import matplotlib
from scipy import stats
matplotlib.use('Agg')

import matplotlib.pyplot as plt 

INPUT_FILE = sys.argv[1]
# INPUT_FILE = '/storage/Nicholas/Data/pixel_classification/pixel_classification_test_LC81990242013280LGN00.csv'
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
df = pd.read_csv(INPUT_FILE, engine='c', dtype=float32_cols, header = header_) 

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
            'fmask',
            'classification',
            'manual_classification',
            ]



if header_ == None:            
    df.columns = my_list_headers

scene_id = sys.argv[2]



def plot_a_thing_toa_fill_between(_thing):
      """
      thing is a string
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

      plt.close('all')
      plt.fill_between(list(range(len(rtoa_thing_max))), rtoa_thing_min.values, rtoa_thing_max.values, facecolor='blue', alpha=0.5)
      plt.plot(list(range(len(rtoa_thing_max))), rtoa_thing_min.values, 'k--')
      plt.plot(list(range(len(rtoa_thing_max))), rtoa_thing_max.values, 'k--')
      plt.plot(list(range(len(rtoa_thing_max))), rtoa_thing_mean.values)
      plt.xticks(list(range(len(rtoa_thing_max))), list(rtoa_thing_mean.keys()))
      plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
      plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_'+_thing+ '_toa'+'.png')

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
      plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
      fig.tight_layout()
      plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_'+_thing+ '_toa_errorbars_manual'+'.png')
      # Removing outliers
      
plot_a_thing_toa_errorbars_manual('cloud', 2)
plot_a_thing_toa_errorbars_manual('land', 0.8)
plot_a_thing_toa_errorbars_manual('water', 0.25)
plot_a_thing_toa_errorbars_manual('shadow', 0.5)

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
      plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
      fig.tight_layout()
      plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_'+classification[_thing]+ '_toa_errorbars_fmask'+'.png')

plot_a_thing_toa_errorbars_fmask(0, 0.8)
plot_a_thing_toa_errorbars_fmask(1, 0.25)
plot_a_thing_toa_errorbars_fmask(2, 0.5)
plot_a_thing_toa_errorbars_fmask(4, 2)

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
      plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
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
      plt.annotate("Created using: " + os.path.realpath(__file__), xycoords='axes fraction', xy=(0, 0), xytext=(-0.1, -0.1), fontsize=8)
      fig.tight_layout()
      plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_'+classification[_thing]+'_'+classification[_thing2]+ '_toa_errorbars_classification'+'.png')

plot_a_thing_toa_errorbars_classification(1, 0.25)
plot_a_thing_toa_errorbars_classification(2, 0.8)
plot_a_thing_toa_errorbars_classification(4, 0.5)
plot_a_thing_toa_errorbars_classification(5, 2)
plot_a_thing_toa_errorbars_classification(6, 1.5)
plot_a_thing_toa_errorbars_classification_two(5, 6, 2)

# clouds = df[df['manual_classification']=='cloud']
# rtoa_clouds = clouds.ix[:,'rtoa_coastal':'rtoa_swir2']
# # rtoa_clouds_real = rtoa_clouds[rtoa_clouds.applymap(np.isreal).all(1)]
# rtoa_clouds_real = rtoa_clouds
# rtoa_clouds_real = rtoa_clouds_real.convert_objects(convert_numeric=True)
# rtoa_clouds_mean = rtoa_clouds_real.mean()
# rtoa_clouds_max = rtoa_clouds_real.max()
# rtoa_clouds_min = rtoa_clouds_real.min()

# plt.close('all')
# plt.fill_between(list(range(len(rtoa_clouds_max))), rtoa_clouds_min.values, rtoa_clouds_max.values, facecolor='blue', alpha=0.5)
# plt.plot(list(range(len(rtoa_clouds_max))), rtoa_clouds_mean.values)
# plt.xticks(list(range(len(rtoa_clouds_max))), list(rtoa_clouds_mean.keys()))
# plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_rtoa_clouds'+'.png')

# land = df[df['manual_classification']=='land']
# rtoa_land = land.ix[:,'rtoa_coastal':'rtoa_swir2']
# # rtoa_land_real = rtoa_land[rtoa_land.applymap(np.isreal).all(1)]
# rtoa_land_real = rtoa_land
# rtoa_land_real = rtoa_land_real.convert_objects(convert_numeric=True)
# rtoa_land_mean = rtoa_land_real.mean()
# rtoa_land_max = rtoa_land_real.max()
# rtoa_land_min = rtoa_land_real.min()

# plt.close('all')
# plt.fill_between(list(range(len(rtoa_land_max))), rtoa_land_min.values, rtoa_land_max.values, facecolor='blue', alpha=0.5)
# plt.plot(list(range(len(rtoa_land_max))), rtoa_land_mean.values)
# plt.xticks(list(range(len(rtoa_land_max))), list(rtoa_land_mean.keys()))
# plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_rtoa_land'+'.png')

# water = df[df['manual_classification']=='water']
# rtoa_water = water.ix[:,'rtoa_coastal':'rtoa_swir2']
# # rtoa_water_real = rtoa_water[rtoa_water.applymap(np.isreal).all(1)]
# rtoa_water_real = rtoa_water
# rtoa_water_real = rtoa_water_real.convert_objects(convert_numeric=True)
# rtoa_water_mean = rtoa_water_real.mean()
# rtoa_water_max = rtoa_water_real.max()
# rtoa_water_min = rtoa_water_real.min()

# plt.close('all')
# plt.fill_between(list(range(len(rtoa_water_max))), rtoa_water_min.values, rtoa_water_max.values, facecolor='blue', alpha=0.5)
# plt.plot(list(range(len(rtoa_water_max))), rtoa_water_mean.values)
# plt.xticks(list(range(len(rtoa_water_max))), list(rtoa_water_mean.keys()))
# plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_rtoa_water'+'.png')

# shadow = df[df['manual_classification']=='shadow']
# rtoa_shadow = shadow.ix[:,'rtoa_coastal':'rtoa_swir2']
# # rtoa_shadow_real = rtoa_shadow[rtoa_shadow.applymap(np.isreal).all(1)]
# rtoa_shadow_real = rtoa_shadow
# rtoa_shadow_real = rtoa_shadow_real.convert_objects(convert_numeric=True)
# rtoa_shadow_mean = rtoa_shadow_real.mean()
# rtoa_shadow_max = rtoa_shadow_real.max()
# rtoa_shadow_min = rtoa_shadow_real.min()

# plt.close('all')
# plt.fill_between(list(range(len(rtoa_shadow_max))), rtoa_shadow_min.values, rtoa_shadow_max.values, facecolor='blue', alpha=0.5)
# plt.plot(list(range(len(rtoa_shadow_max))), rtoa_shadow_mean.values)
# plt.xticks(list(range(len(rtoa_shadow_max))), list(rtoa_shadow_mean.keys()))
# plt.savefig('/storage/Nicholas/Data/pixel_classification/'+scene_id+'_rtoa_shadow'+'.png')