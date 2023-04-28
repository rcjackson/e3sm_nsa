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
    heights = ds["Heights"]
    first_time = ds['base_time']
    
    output_ds = {}
    
    mpc_occurrence = xr.where(np.logical_and(ds["Avg_Retrieved_LWC"] > 1e-4, ds["Avg_Retrieved_IWC"] > 1e-1), 1, 0).sum(
        dim='time').values[np.newaxis, :]
    num_clouds = xr.where(np.logical_or(ds["Avg_Retrieved_LWC"] > 1e-4, ds["Avg_Retrieved_IWC"] > 1e-1), 1, 0).sum(dim='time').values[np.newaxis, :]
    mpc_occurrence = mpc_occurrence / num_clouds
    mpc_occurrence = xr.DataArray(mpc_occurrence, dims=('time', 'nheights'))
    mpc_occurrence.attrs["long_name"] = "Mixed phase cloud occurrence"
    mpc_occurrence.attrs["units"] = "1"
    output_ds["mpc_occurrence"] = mpc_occurrence
    # Do mean and confidence interval calculations
    for var in ds.variables:
        if ds[var].dims == ('time', 'nheights') and not var == "CloudPhaseMask":
            if var == "CloudRetrievalMask":
                continue
            bootstrap_means = np.nanmean(ds[var].values, axis=0)
            bootstrap_intervals = np.zeros((1, len(bootstrap_means), 2))            
            
            booty = bootstrap((ds[var].values,), np.nanmean, method='percentile', axis=0)
            bootstrap_intervals[0, :, 0] = booty.confidence_interval.low
            bootstrap_intervals[0, :, 1] = booty.confidence_interval.high    

            # Store the results in an xarray Dataset with metadata
            daily_mean = xr.DataArray(bootstrap_means[np.newaxis, :], dims=('time', 'nheights'))
            daily_mean.attrs["long_name"] = "Daily mean of %s" % ds[var].attrs["long_name"]
            daily_mean.attrs["units"] = ds[var].attrs["units"]
            output_ds[var] = daily_mean
            confidence = xr.DataArray(bootstrap_intervals,
                dims=('time', 'nheights', 'interval'))
            confidence.attrs["long_name"] = "95 percent Confidence interval of %s" % ds[var].attrs["long_name"]
            confidence.attrs["units"] = ds[var].attrs["units"]
            output_ds[var + '_confidence'] = confidence
        elif var == "CloudPhaseMask":
            data = ds[var].values
            
            def pct_clear(x, axis=0):
                return phase_fraction(x, [0], axis)
 
            def pct_ice(x, axis=0):
                return phase_fraction(x, [1], axis)

            def pct_liquid(x, axis=0):
                return phase_fraction(x, [3, 5, 8], axis)

            def pct_snow(x, axis=0):
                return phase_fraction(x, [2], axis)
 
            def pct_rain(x, axis=0):
                return phase_fraction(x, [6], axis)

            def pct_mixed(x, axis=0):
                return phase_fraction(x, [7], axis)
            
            # Count clear times
            clear_pct = pct_clear((data,))
            bootstrap_intervals = np.zeros((1, len(clear_pct), 2))
            booty = bootstrap((data,), pct_clear, method='percentile', axis=0)
            daily_mean = xr.DataArray(clear_pct[np.newaxis, :], dims=('time', 'nheights'))
            daily_mean.attrs["long_name"] = "Daily mean of %s" % ds[var].attrs["long_name"]
            daily_mean.attrs["units"] = "1"
            output_ds['pct_clear'] = daily_mean
            bootstrap_intervals[0, :, 0] = booty.confidence_interval.low
            bootstrap_intervals[0, :, 1] = booty.confidence_interval.high
            confidence = xr.DataArray(bootstrap_intervals[:, :],
                    dims=('time', 'nheights', 'interval'))
            confidence.attrs["long_name"] = "95 percent Confidence interval of clear air percentage"
            confidence.attrs["units"] = "1"
            output_ds['clear_confidence'] = confidence

            # Count ice times
            ice_pct = pct_ice((data,))
            bootstrap_intervals = np.zeros((1, len(clear_pct), 2))
            booty = bootstrap((data,), pct_ice, method='percentile', axis=0)
            daily_mean = xr.DataArray(ice_pct[np.newaxis, :], dims=('time', 'nheights'))
            daily_mean.attrs["long_name"] = "Fraction of time spent in ice cloud"
            daily_mean.attrs["units"] = "1"
            output_ds['pct_ice'] = daily_mean
            bootstrap_intervals[0, :, 0] = booty.confidence_interval.low
            bootstrap_intervals[0, :, 1] = booty.confidence_interval.high
            confidence = xr.DataArray(bootstrap_intervals[:, :], 
                dims=('time', 'nheights', 'interval'))
            confidence.attrs["long_name"] = "Fraction of time spent in ice confidence interval"
            confidence.attrs["units"] = "1"
            output_ds['ice_confidence'] = confidence
                    
            # Count liquid times
            liquid_pct = pct_liquid((data,))
            bootstrap_intervals = np.zeros((1, len(clear_pct), 2))
            booty = bootstrap((data,), pct_liquid, method='percentile', axis=0)
            daily_mean = xr.DataArray(liquid_pct[np.newaxis, :], dims=('time', 'nheights'))
            daily_mean.attrs["long_name"] = "Fraction of time spent in liquid cloud"
            daily_mean.attrs["units"] = "1"
            output_ds['pct_liquid'] = daily_mean
            bootstrap_intervals[0, :, 0] = booty.confidence_interval.low
            bootstrap_intervals[0, :, 1] = booty.confidence_interval.high
            confidence = xr.DataArray(bootstrap_intervals[:, :],
                dims=('time', 'nheights', 'interval'))
            confidence.attrs["long_name"] = "Fraction of time spent in liquid confidence interval"
            confidence.attrs["units"] = "1"
            output_ds['liquid_confidence'] = confidence
 
            # Count rain times
            rain_pct = pct_rain((data,))
            bootstrap_intervals = np.zeros((1, len(clear_pct), 2))
            booty = bootstrap((data,), pct_rain, method='percentile', axis=0)
            daily_mean = xr.DataArray(rain_pct[np.newaxis, :], dims=('time', 'nheights'))
            daily_mean.attrs["long_name"] = "Fraction of time spent in rain cloud"
            daily_mean.attrs["units"] = "1"
            output_ds['pct_rain'] = daily_mean
            bootstrap_intervals[0, :, 0] = booty.confidence_interval.low
            bootstrap_intervals[0, :, 1] = booty.confidence_interval.high
            confidence = xr.DataArray(bootstrap_intervals[:, :],
                dims=('time', 'nheights', 'interval'))
            confidence.attrs["long_name"] = "Fraction of time spent in rain confidence interval"
            confidence.attrs["units"] = "1"
            output_ds['rain_confidence'] = confidence

            # Count snow times
            snow_pct = pct_snow((data,))
            bootstrap_intervals = np.zeros((1, len(clear_pct), 2))
            booty = bootstrap((data,), pct_snow, method='percentile', axis=0)
            daily_mean = xr.DataArray(snow_pct[np.newaxis, :], dims=('time', 'nheights'))
            daily_mean.attrs["long_name"] = "Fraction of time spent in snow cloud"
            daily_mean.attrs["units"] = "1"
            output_ds['pct_snow'] = daily_mean
            bootstrap_intervals[0, :, 0] = booty.confidence_interval.low
            bootstrap_intervals[0, :, 1] = booty.confidence_interval.high
            confidence = xr.DataArray(bootstrap_intervals[:, :],
                dims=('time', 'nheights', 'interval'))
            confidence.attrs["long_name"] = "Fraction of time spent in snow confidence interval"
            confidence.attrs["units"] = "1"
            output_ds['snow_confidence'] = confidence

            # Count mixed times
            mixed_pct = pct_mixed((data,))
            bootstrap_intervals = np.zeros((1, len(clear_pct), 2))
            booty = bootstrap((data, ), pct_mixed, method='percentile', axis=0)
            daily_mean = xr.DataArray(mixed_pct[np.newaxis, :], dims=('time', 'nheights'))
            daily_mean.attrs["long_name"] = "Fraction of time spent in mixed cloud"
            daily_mean.attrs["units"] = "1"
            output_ds['pct_mixed'] = daily_mean
            bootstrap_intervals[0, :, 0] = booty.confidence_interval.low
            bootstrap_intervals[0, :, 1] = booty.confidence_interval.high
            confidence = xr.DataArray(bootstrap_intervals[:, :],
                dims=('time', 'nheights', 'interval'))
            confidence.attrs["long_name"] = "Fraction of time spent in mixed confidence interval"
            confidence.attrs["units"] = "1"
            output_ds['mixed_confidence'] = confidence

    print(list(output_ds.keys()))
    out_ds = xr.Dataset(output_ds, coords={'time': np.atleast_1d(first_time),
        'nheights': ds['nheights']})
    print(out_ds)
    ds.close()
    return out_ds

# Get the list of microbase files
if __name__ == "__main__":
    date_str = '%02d%02d' % (int(sys.argv[1]), int(sys.argv[2]))
    in_path = '/lcrc/group/earthscience/rjackson/nsa_microbase/st/*%s*.cdf' % date_str
    file_list = glob.glob(in_path)
    results = list(map(do_bootstrapping, file_list))
    #bag = db.from_sequence(file_list)
    #results = bag.map(do_bootstrapping).compute()
    out_ds = xr.concat(results, dim='time').sortby('time')
    out_ds.to_netcdf('daily_product/microbase_%s.nc' % date_str)
