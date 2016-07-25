# afar

To use AFAR

Using python 3 and assuming dependencies are met see `txt_files/requirements.txt`

If you have miniconda installed you can (in the afar directory):

    $ conda create --name afar_env python=3 
    $ source activate afar_env
    $ conda install –file ./txt_files/requirements.txt

    $ python afar.py /path/to/directory_with_toa_netcdfs SCENEID /path/to/output_directory

or 

    $ bash bulk_afar.sh ./txt_files/LC81990242013280LGN00.txt for example…

and then you should get…

File written to: `/storage/Nicholas/Data/20151224/LC81990242013280LGN00_final_mask.nc`
