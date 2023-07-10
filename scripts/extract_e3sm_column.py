import xarray as xr
import glob
import os
import sys

e3sm_list = '/lcrc/globalscratch/yfeng/NSA/Data/%s/*.nc' % sys.argv[1]
out_path = '/lcrc/group/earthscience/rjackson/e3sm_nsa/data/e3sm_data'

nsa_lat = 71 + 19/60 + 22.8/3600
nsa_lon = -156 - 36/60 - 54/3600 + 360. 

file_list = glob.glob(e3sm_list)

for fi in file_list:
    print(fi)
    base, name = os.path.split(fi)
    try:
        input_ds = xr.open_dataset(fi)
    except:
        print('%s not processed' % fi)
        continue
    one_column_only = input_ds.sel(lat=nsa_lat, lon=nsa_lon, method='nearest')
    one_column_only.to_netcdf(os.path.join(out_path, name))
    input_ds.close()
    one_column_only.close()


