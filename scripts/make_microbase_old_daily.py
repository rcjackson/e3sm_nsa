import dask.bag as db
import xarray as xr
import glob
import numpy as np
import sys

from scipy.stats import bootstrap
from distributed import LocalCluster, Client

def phase_fraction(data, phase=[0], axis=0, num_points=None):
    phases = np.zeros(data[0].shape[1])
    for k in range(data[0].shape[1]):
        phases[k] = len(np.argwhere(np.isin(data[0][:, k], phase))) / data[0].shape[0]
    return phases

# Get the bootstrapped statistics of the 24 hour average
def do_bootstrapping(file_name):
    print(file_name)
    ds = xr.open_dataset(file_name, chunks=None)
    ds.load()
    try:
        heights = ds["Heights"]
    except KeyError:
        heights = ds["heights"]
    first_time = ds['base_time']
    
    output_ds = {}
    # Move everything to lower case for consistency
    if ds.time[1].dt.year >= 2009:
        lower_case_dict = {}
        for var in ds.variables:
            lower_case_dict[var] = var.lower()
        ds = ds.rename(lower_case_dict)

    output_ds["heights"] = heights
    
    mpc_occurrence = xr.where(np.logical_and(ds["avg_retrieved_lwc"] > 1e-4, ds["avg_retrieved_iwc"] > 1e-1), 1, 0).sum(
        dim='time').values[np.newaxis, :]
    num_clouds = xr.where(
        np.logical_or(ds["avg_retrieved_lwc"] > 1e-4,
                      ds["avg_retrieved_iwc"] > 1e-1), 1, 0).sum(dim='time').values[np.newaxis, :]
    total_times = ds["avg_retrieved_lwc"].shape[0]
    mpc_occurrence = mpc_occurrence / num_clouds
    pct_clear = 1 - num_clouds / total_times
    pct_clear = xr.DataArray(pct_clear, dims=('time', 'nheights'))
    pct_clear.attrs["long_name"] = "Percent clear"
    pct_clear.attrs["units"] = "1"
    mpc_occurrence = xr.DataArray(mpc_occurrence, dims=('time', 'nheights'))
    mpc_occurrence.attrs["long_name"] = "Mixed phase cloud occurrence"
    mpc_occurrence.attrs["units"] = "1"
    output_ds["mpc_occurrence"] = mpc_occurrence
    output_ds["pct_clear"] = pct_clear
    for var in ds.variables:
        if ds[var].dims == ('time', 'nheights') or ds[var].dims == ('time', 'heights'):
            if var == "CloudRetrievalMask":
                continue
            
            bootstrap_means = ds[var].mean(dim='time', skipna=True).values
            # Store the results in an xarray Dataset with metadata
            daily_mean = xr.DataArray(bootstrap_means[np.newaxis, :], dims=('time', 'nheights'))
            daily_mean.attrs["long_name"] = "Daily mean of %s" % ds[var].attrs["long_name"]
            daily_mean.attrs["units"] = ds[var].attrs["units"]
            output_ds[var] = daily_mean

    out_ds = xr.Dataset(output_ds, coords={'time': np.atleast_1d(first_time)})
    out_ds = out_ds.set_coords(['time', 'heights'])
    print(out_ds)
    ds.close()
    return out_ds

# Get the list of microbase files
if __name__ == "__main__":
    date_str = '%02d%02d' % (int(sys.argv[1]), int(sys.argv[2]))
    in_path = '/lcrc/group/earthscience/rjackson/nsa_microbase/*%s*.cdf' % date_str
    file_list = glob.glob(in_path)
    results = list(map(do_bootstrapping, file_list))
    #bag = db.from_sequence(file_list)
    #results = bag.map(do_bootstrapping).compute()
    out_ds = xr.concat(results, dim='time').sortby('time')
    out_ds.to_netcdf('microbase_old_daily/microbase_%s.nc' % date_str)




