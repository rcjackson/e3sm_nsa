from matplotlib import use 
use('agg')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cftime

from scipy.interpolate import interp1d
# Point to the paths of our data

e3sm_path = '/lcrc/group/earthscience/rjackson/e3sm_nsa/data/e3sm_data/v2_CTL.ne30pg2_15yrs.*'
ps_ds_path = '/lcrc/group/earthscience/rjackson/e3sm_nsa/data/e3sm_data/PS*.nc'

nsa_data_path = '/lcrc/group/earthscience/NSA/Johannes-ck-new.dat'

# Load data
nsa_cluster = pd.read_csv(nsa_data_path, index_col=["time"], parse_dates=True)
e3sm_avgs = xr.open_mfdataset(e3sm_path + '*.nc')
ps_ds = xr.open_mfdataset(ps_ds_path)
Rd = 287.057
g0 = 9.80665
e3sm_avgs["P"] = e3sm_avgs["hyam"] * e3sm_avgs["P0"] + e3sm_avgs["hybm"] * ps_ds["PS"]
e3sm_avgs["heights"] = Rd * e3sm_avgs["T"] * (-np.log10(e3sm_avgs["P"]) + np.log10(ps_ds["PS"])) / g0 * 1e-3
print(e3sm_avgs)
rho = e3sm_avgs["P"] / (Rd * e3sm_avgs["T"])
content_conversion_factor = rho * 1e3

# Reindex cluster data to e3sm statistics
tolerance = np.timedelta64(1, 'D') / np.timedelta64(1, 'ns')
nsa_cluster = nsa_cluster.to_xarray().sortby('time')
nsa_cluster = nsa_cluster.convert_calendar('noleap')
nsa_cluster = nsa_cluster.reindex(time=e3sm_avgs.time, method='nearest')

print(nsa_cluster["class"])
print(e3sm_avgs)
e3sm_avgs["cluster"] = nsa_cluster["class"]
print(e3sm_avgs["CLDLIQ"].values.shape)
print(e3sm_avgs["CLDICE"].values.shape)
print(e3sm_avgs["CLOUD"].values.shape)
heights = np.arange(0, 10., 0.01)

e3sm_avgs["CLOUD_FILTER"] = e3sm_avgs["CLOUD"].where(e3sm_avgs["CLOUD"] > 0.01)
LWC_array = e3sm_avgs["CLDLIQ"].values / e3sm_avgs["CLOUD_FILTER"].values * content_conversion_factor
IWC_array = e3sm_avgs["CLDICE"].values / e3sm_avgs["CLOUD_FILTER"].values * content_conversion_factor
LWC_array += e3sm_avgs["RAINQM"].values * content_conversion_factor
IWC_array += e3sm_avgs["SNOWQM"].values * content_conversion_factor
LWC_array = np.where(np.isfinite(LWC_array), LWC_array, 0)
IWC_array = np.where(np.isfinite(IWC_array), IWC_array, 0)
#LWC_array[~np.isfinite(LWC_array)] = 0
#IWC_array[~np.isfinite(IWC_array)] = 0
clear_pct = 1 - e3sm_avgs["CLOUD"].values
height = e3sm_avgs["heights"].values

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

e3sm_avgs["Avg_Retrieved_LWC"] = xr.DataArray(
    data=LWC_array_interp,
    dims=["time", "height"],
    coords={"time": e3sm_avgs.time.values, "height": heights})

e3sm_avgs["Avg_Retrieved_IWC"] = xr.DataArray(
    data=IWC_array_interp,
    dims=["time", "height"],
    coords={"time": e3sm_avgs.time.values, "height": heights})

e3sm_avgs["pct_clear"] = xr.DataArray(data=clear_array_interp,
    dims=["time", "height"],
    coords={"time": e3sm_avgs.time.values, "height": heights})

e3sm_avgs["Avg_Retrieved_LWC"] = e3sm_avgs["Avg_Retrieved_LWC"].where(
   e3sm_avgs["Avg_Retrieved_LWC"] > 1e-4)
e3sm_avgs["Avg_Retrieved_IWC"] = e3sm_avgs["Avg_Retrieved_IWC"].where(
   e3sm_avgs["Avg_Retrieved_IWC"] > 1e-4)
e3sm_avgs["mpc_occurrence"] = xr.where(np.logical_and(
   e3sm_avgs["Avg_Retrieved_IWC"] > 1e-4, e3sm_avgs["Avg_Retrieved_LWC"] > 1e-4), 1 - e3sm_avgs["pct_clear"], 0)
                                        
e3sm_avgs["pct_clear"] = xr.DataArray(data=clear_array_interp,
    dims=["time", "height"],
    coords={"time": e3sm_avgs.time.values, "height": heights}) 
e3sm_avgs["season"] = e3sm_avgs["time"].dt.season

e3sm_groupby = e3sm_avgs.groupby("time.season").mean(skipna=True)
print(e3sm_groupby)
fig, ax = plt.subplots(4, 4, figsize=(22, 20))
colors = ['b', 'g', 'k', 'c']
i = 0
def cluster_mean(x, cluster_no):
    return x.where(x.cluster == cluster_no).mean("time", skipna=True)

for cluster in [1, 2, 3, 4]:
    cmean = lambda x: cluster_mean(x, cluster)
    e3sm_groupby = e3sm_avgs.groupby("season").apply(cmean)
    e3sm_groupby = e3sm_groupby.set_coords(["season", "height"])
    
    print(e3sm_groupby)
    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        e3sm_groupby['Avg_Retrieved_LWC'].sel(season=season).T.plot(
            ax=ax[i, 0], y="height", label=str(cluster), color=colors[cluster - 1])

        ax[i, 0].set_xlabel('LWC [g $m^{-3}$]')
        ax[i, 0].set_ylabel('Height [km]')
        ax[i ,0].set_title('')
        ax[i, 0].set_xlim([0, 0.1])
        ax[i, 0].set_ylim([0, 10])
        ax[i, 0].set_title(season)
        ax[i, 0].legend()
    
        e3sm_groupby['Avg_Retrieved_IWC'].sel(season=season).T.plot(
            ax=ax[i, 1], y="height", label=str(cluster), color=colors[cluster - 1])
        ax[i, 1].set_xlabel('IWC [g $m^{-3}$]')
        ax[i, 1].set_ylabel('Height [km]')
        ax[i, 1].set_title('')
        ax[i, 1].set_ylim([0, 10])
        ax[i, 1].set_xlim([0, 0.03])
        ax[i, 1].set_title(season)

        (1 - e3sm_groupby['pct_clear']).sel(season=season).T.plot(
            ax=ax[i, 2], y="height", label=str(cluster), color=colors[cluster - 1])
        ax[i, 2].set_xlabel('Cloud fraction')
        ax[i, 2].set_ylabel('Height [km]')
        ax[i, 2].set_title('')
        ax[i, 2].set_xlim([0, 1])
        ax[i, 2].set_ylim([0, 10])
        ax[i, 2].set_title(season)
        
        e3sm_groupby['mpc_occurrence'].sel(season=season).T.plot(
            ax=ax[i, 3], y="height", label=str(cluster), color=colors[cluster - 1])
        ax[i, 3].set_xlabel('MPC occurrence')
        ax[i, 3].set_ylabel('Height [km]')
        ax[i, 3].set_title('')
        ax[i, 3].set_ylim([0, 10])
        ax[i, 3].set_xlim([0, 1])
        ax[i, 3].set_title(season)


        
fig.savefig('mean_stats.png', bbox_inches='tight')
print(e3sm_groupby['Avg_Retrieved_LWC'].values)
e3sm_avgs.close()
nsa_cluster.close()
