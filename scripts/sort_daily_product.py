import xarray as xr
import glob
import os

from copy import deepcopy
file_list = glob.glob('daily_product/*.nc')
for fi in file_list:
    print(fi)
    base, name = os.path.split(fi)
    with xr.open_dataset(fi) as ds:
        print(ds)
        sorted_ds = deepcopy(ds.sortby('time'))
    sorted_ds.to_netcdf(os.path.join('sorted_daily_product', name))
