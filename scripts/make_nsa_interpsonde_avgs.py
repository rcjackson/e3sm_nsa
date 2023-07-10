import xarray as xr

from distributed import Client, LocalCluster

if __name__ == "__main__":
    nsa_interpsonde = '/lcrc/group/earthscience/rjackson/nsainterpsonde/24hr_out/nsainterpolatedsondeC1.c1.*%04d*.nc'
    for year in range(2003, 2021):
       try:
           ds = xr.open_mfdataset(nsa_interpsonde % (year), parallel=True)
           ds.to_netcdf('/lcrc/group/earthscience/rjackson/nsainterpsonde/24hr_out/nsainterpsonde_%04d.nc' % (year))
       except OSError:
           continue
