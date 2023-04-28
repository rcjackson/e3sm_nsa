from matplotlib import use 
use('agg')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cftime

from scipy.interpolate import interp1d
# Point to the paths of our data

microbase_path = '/lcrc/group/earthscience/rjackson/e3sm_nsa/data/e3sm_data/v2_CTL.ne30pg2_15yrs.*'
ps_ds_path = '/lcrc/group/earthscience/rjackson/e3sm_nsa/data/e3sm_data/PS*.nc'

nsa_data_path = '/lcrc/group/earthscience/NSA/Johannes-ck-new.dat'
full_file_path = '/lcrc/group/earthscience/rjackson/nsa_microbase/st/nsamicrobase2shupeturnC1.c1.20191027.001000.cdf'

# Load data
nsa_cluster = pd.read_csv(nsa_data_path, index_col=["time"], parse_dates=True)
microbase_avgs = xr.open_mfdataset(microbase_path + '*.nc')
ps_ds = xr.open_mfdataset(ps_ds_path)
Rd = 287.057
g0 = 9.80665
microbase_avgs["P"] = microbase_avgs["hyam"] * microbase_avgs["P0"] + microbase_avgs["hybm"] * ps_ds["PS"]
microbase_avgs["heights"] = Rd * microbase_avgs["T"] * (-np.log10(microbase_avgs["P"]) + np.log10(ps_ds["PS"])) / g0 * 1e-3
print(microbase_avgs)
rho = microbase_avgs["P"] / (Rd * microbase_avgs["T"])
content_conversion_factor = rho * 1e3

# Reindex cluster data to microbase statistics
tolerance = np.timedelta64(1, 'D') / np.timedelta64(1, 'ns')
nsa_cluster = nsa_cluster.to_xarray().sortby('time')
nsa_cluster = nsa_cluster.convert_calendar('noleap')
nsa_cluster = nsa_cluster.reindex(time=microbase_avgs.time, method='nearest')

print(nsa_cluster["class"])
print(microbase_avgs)
microbase_avgs["cluster"] = nsa_cluster["class"]
print(microbase_avgs["CLDLIQ"].values.shape)
print(microbase_avgs["CLDICE"].values.shape)
print(microbase_avgs["CLOUD"].values.shape)
heights = np.arange(0, 10., 0.01)

microbase_avgs["CLOUD_FILTER"] = microbase_avgs["CLOUD"].where(microbase_avgs["CLOUD"] > 0.01)
LWC_array = microbase_avgs["CLDLIQ"].values / microbase_avgs["CLOUD_FILTER"].values * content_conversion_factor
IWC_array = microbase_avgs["CLDICE"].values / microbase_avgs["CLOUD_FILTER"].values * content_conversion_factor
LWC_array = np.where(np.isfinite(LWC_array), LWC_array, 0)
IWC_array = np.where(np.isfinite(IWC_array), IWC_array, 0)
#LWC_array[~np.isfinite(LWC_array)] = 0
#IWC_array[~np.isfinite(IWC_array)] = 0
clear_pct = 1 - microbase_avgs["CLOUD"].values
height = microbase_avgs["heights"].values

LWC_array_interp = np.zeros((LWC_array.shape[0], len(heights)))
IWC_array_interp = np.zeros((LWC_array.shape[0], len(heights)))
clear_array_interp = np.zeros((LWC_array.shape[0], len(heights)))
for i in range(LWC_array.shape[0]):
    if i % 100 == 0:
        print('%d/%d' % (i, LWC_array.shape[0]))
    LWC = interp1d(height[i, :], LWC_array[i, :], fill_value="extrapolate")
    IWC = interp1d(height[i, :], IWC_array[i, :], fill_value="extrapolate")
    clear = interp1d(height[i, :], clear_pct[i, :], fill_value="extrapolate")
    LWC_array_interp[i, :] = LWC(heights)
    IWC_array_interp[i, :] = IWC(heights)
    clear_array_interp[i, :] = clear(heights)

microbase_avgs["Avg_Retrieved_LWC"] = xr.DataArray(
    data=LWC_array_interp,
    dims=["time", "height"],
    coords={"time": microbase_avgs.time.values, "height": heights})

microbase_avgs["Avg_Retrieved_IWC"] = xr.DataArray(
    data=IWC_array_interp,
    dims=["time", "height"],
    coords={"time": microbase_avgs.time.values, "height": heights})

microbase_avgs["pct_clear"] = xr.DataArray(data=clear_array_interp,
    dims=["time", "height"],
    coords={"time": microbase_avgs.time.values, "height": heights})

microbase_avgs["Avg_Retrieved_LWC"] = microbase_avgs["Avg_Retrieved_LWC"].where(
   microbase_avgs["Avg_Retrieved_LWC"] > 1e-4)
microbase_avgs["Avg_Retrieved_IWC"] = microbase_avgs["Avg_Retrieved_IWC"].where(
   microbase_avgs["Avg_Retrieved_IWC"] > 1e-4)
microbase_avgs["mpc_occurrence"] = xr.where(np.logical_and(
   microbase_avgs["Avg_Retrieved_IWC"] > 1e-4, microbase_avgs["Avg_Retrieved_LWC"] > 1e-4), 1 - microbase_avgs["pct_clear"], 0)
                                        
microbase_avgs["pct_clear"] = xr.DataArray(data=clear_array_interp,
    dims=["time", "height"],
    coords={"time": microbase_avgs.time.values, "height": heights}) 
microbase_avgs["season"] = microbase_avgs["time"].dt.season

#microbase_avgs = microbase_avgs.interp(lev=heights)
#micro_multiindex = microbase_avgs.stack(my_multiindex=['time.season', 'cluster'])
microbase_groupby = microbase_avgs.groupby("time.season").mean(skipna=True)
print(microbase_groupby)
fig, ax = plt.subplots(4, 4, figsize=(22, 20))
colors = ['b', 'g', 'k', 'c']
i = 0
def cluster_mean(x, cluster_no):
    return x.where(x.cluster == cluster_no).mean("time", skipna=True)

for cluster in [1, 2, 3, 4]:
    cmean = lambda x: cluster_mean(x, cluster)
    microbase_groupby = microbase_avgs.groupby("season").apply(cmean)
    microbase_groupby = microbase_groupby.set_coords(["season", "height"])
    
    print(microbase_groupby)
    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        microbase_groupby['Avg_Retrieved_LWC'].sel(season=season).T.plot(
            ax=ax[i, 0], y="height", label=str(cluster), color=colors[cluster - 1])

        ax[i, 0].set_xlabel('LWC [g $m^{-3}$]')
        ax[i, 0].set_ylabel('Height [km]')
        ax[i ,0].set_title('')
        ax[i, 0].set_xlim([0, 0.1])
        ax[i, 0].set_ylim([0, 10])
        ax[i, 0].set_title(season)
        ax[i, 0].legend()
    
        microbase_groupby['Avg_Retrieved_IWC'].sel(season=season).T.plot(
            ax=ax[i, 1], y="height", label=str(cluster), color=colors[cluster - 1])
        ax[i, 1].set_xlabel('IWC [g $m^{-3}$]')
        ax[i, 1].set_ylabel('Height [km]')
        ax[i, 1].set_title('')
        ax[i, 1].set_ylim([0, 10])
        ax[i, 1].set_xlim([0, 0.03])
        ax[i, 1].set_title(season)

        (1 - microbase_groupby['pct_clear']).sel(season=season).T.plot(
            ax=ax[i, 2], y="height", label=str(cluster), color=colors[cluster - 1])
        ax[i, 2].set_xlabel('Cloud fraction')
        ax[i, 2].set_ylabel('Height [km]')
        ax[i, 2].set_title('')
        ax[i, 2].set_xlim([0, 1])
        ax[i, 2].set_ylim([0, 10])
        ax[i, 2].set_title(season)
        
        microbase_groupby['mpc_occurrence'].sel(season=season).T.plot(
            ax=ax[i, 3], y="height", label=str(cluster), color=colors[cluster - 1])
        ax[i, 3].set_xlabel('MPC occurrence')
        ax[i, 3].set_ylabel('Height [km]')
        ax[i, 3].set_title('')
        ax[i, 3].set_ylim([0, 10])
        ax[i, 3].set_xlim([0, 1])
        ax[i, 3].set_title(season)


        
fig.savefig('mean_stats.png', bbox_inches='tight')
print(microbase_groupby['Avg_Retrieved_LWC'].values)
microbase_avgs.close()
nsa_cluster.close()
