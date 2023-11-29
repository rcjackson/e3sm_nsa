from matplotlib import use 
use('agg')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Point to the paths of our data
microbase_path = '/lcrc/group/earthscience/rjackson/e3sm_nsa/scripts/daily_product/'
nsa_data_path = '/lcrc/group/earthscience/NSA/Johannes-ck-new.dat'
full_file_path = '/lcrc/group/earthscience/rjackson/nsa_microbase/st/nsamicrobase2shupeturnC1.c1.20191027.001000.cdf'

# Load data
nsa_cluster = pd.read_csv(nsa_data_path, index_col=["time"], parse_dates=True)
microbase_avgs = xr.open_mfdataset(microbase_path + '*.nc')
print(microbase_avgs)

# Reindex cluster data to microbase statistics
nsa_cluster = nsa_cluster.to_xarray().sortby('time')
tolerance = np.timedelta64(1, 'D') / np.timedelta64(1, 'ns')
microbase_avgs = microbase_avgs.reindex(time=nsa_cluster.time, method='nearest',
                                        tolerance=tolerance)

ds = xr.open_dataset(full_file_path)
print(nsa_cluster["class"])
print(microbase_avgs)
microbase_avgs["cluster"] = nsa_cluster["class"]
microbase_avgs["mpc_occurrence"] = microbase_avgs["mpc_occurrence"]
microbase_avgs["nheights"] = ds["Heights"] / 1000
microbase_avgs["pct_mixed"] = microbase_avgs["pct_mixed"]
microbase_avgs["pct_ice"] = microbase_avgs["pct_ice"]
microbase_avgs["pct_liquid"] = microbase_avgs["pct_liquid"]
microbase_avgs["which_phase"] = xr.DataArray(np.argmax(np.stack([microbase_avgs["pct_mixed"], 
                                                                 microbase_avgs["pct_ice"] + microbase_avgs["pct_snow"], 
                                                                 microbase_avgs["pct_liquid"] + microbase_avgs["pct_rain"],
                                                                 microbase_avgs["pct_clear"]]),
                                          axis=0), dims=microbase_avgs["pct_liquid"].dims)


microbase_avgs["which_phase"] = microbase_avgs["which_phase"].where(np.isfinite(microbase_avgs["pct_mixed"]), drop=False)
#microbase_avgs["which_phase"] = microbase_avgs["which_phase"].where(np.logical_and(np.isfinite(microbase_avgs["pct_mixed"]), 
#                                                                                   microbase_avgs["which_phase"] < 3), drop=True)
microbase_avgs["is_mixed"] = xr.where(microbase_avgs["which_phase"] == 0, 100, 0)
microbase_avgs["is_ice"] = xr.where(microbase_avgs["which_phase"] == 1, 100, 0)
microbase_avgs["is_liquid"] = xr.where(microbase_avgs["which_phase"] == 2, 100, 0)
microbase_avgs["is_clear"] = xr.where(microbase_avgs["which_phase"] == 3, 100, 0)
microbase_avgs["Avg_Retrieved_LWC"] = microbase_avgs["Avg_Retrieved_LWC"].where(microbase_avgs.pct_clear < 0.95)
microbase_avgs["Avg_Retrieved_IWC"] = microbase_avgs["Avg_Retrieved_IWC"].where(microbase_avgs.pct_clear < 0.95)
microbase_avgs["Avg_Retrieved_IWC"] = microbase_avgs["Avg_Retrieved_IWC"].where(microbase_avgs.Avg_Retrieved_IWC >= 0.001)
microbase_avgs["Avg_Retrieved_IWC_confidence"] = microbase_avgs["Avg_Retrieved_IWC_confidence"].where(microbase_avgs.Avg_Retrieved_IWC_confidence >= 0.001) / 1e3
microbase_avgs["Avg_Retrieved_IWC"] = microbase_avgs["Avg_Retrieved_IWC"] / 1e3
microbase_avgs["season"] = microbase_avgs["time"].dt.season
#micro_multiindex = microbase_avgs.stack(my_multiindex=['time.season', 'cluster'])
microbase_groupby = microbase_avgs.groupby("time.season").mean(skipna=True)
print(microbase_groupby)
ds.close()
fig, ax = plt.subplots(4, 5, figsize=(12, 12))
colors = ['b', 'g', 'k', 'r']
i = 0
def cluster_mean(x, cluster_no):
    return x.where(x.cluster == cluster_no).mean("time", skipna=True)

def cluster_sum(x, cluster_no):
    return x.where(x.cluster == cluster_no).sum("time", skipna=True)

for cluster in [1, 2, 3, 4]:
    cmean = lambda x: cluster_mean(x, cluster)
    csum = lambda x: cluster_sum(x, cluster)
    microbase_groupby = microbase_avgs.groupby("season").apply(cmean)
     
    print(microbase_groupby)
    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        microbase_groupby['Avg_Retrieved_LWC'].sel(season=season).T.plot(
            y="nheights", ax=ax[i, 0], label=str(cluster), linewidth=2, color=colors[cluster - 1])

        ax[i, 0].set_xlabel('LWC [g $m^{-3}$]')
        ax[i, 0].set_ylabel('Height [km]')
        ax[i ,0].set_title('')
        ax[i, 0].set_xlim([0, 0.1])
        ax[i, 0].set_ylim([0, 10])
        ax[i, 0].set_title(season)
        ax[i, 0].legend()
    
        microbase_groupby['is_liquid'].sel(season=season).T.plot(
            y="nheights", ax=ax[i, 1], label=str(cluster), linewidth=2, color=colors[cluster - 1])
        ax[i, 1].set_xlabel('% of days liquid-dominated')
        ax[i, 1].set_ylabel('Height [km]')
        ax[i, 1].set_title('')
        ax[i, 1].set_ylim([0, 10])
        ax[i, 1].set_xlim([0, 100])
        ax[i, 1].set_title(season)
        
        microbase_groupby['is_ice'].sel(season=season).T.plot(
            y="nheights", ax=ax[i, 2], label=str(cluster), linewidth=2, color=colors[cluster - 1])
        ax[i, 2].set_xlabel('% of days ice-dominated')
        ax[i, 2].set_ylabel('Height [km]')
        ax[i, 2].set_title('')
        ax[i, 2].set_ylim([0, 10])
        ax[i, 2].set_xlim([0, 100])
        ax[i, 2].set_title(season)
        
        microbase_groupby['is_mixed'].sel(season=season).T.plot(
            y="nheights", ax=ax[i, 3], label=str(cluster), linewidth=2, color=colors[cluster - 1])
        ax[i, 3].set_xlabel('% of days mixed-dominated')
        ax[i, 3].set_ylabel('Height [km]')
        ax[i, 3].set_title('')
        ax[i, 3].set_ylim([0, 10])
        ax[i, 3].set_xlim([0, 100])
        ax[i, 3].set_title(season)
        microbase_groupby['is_clear'].sel(season=season).T.plot(
            y="nheights", ax=ax[i, 4], label=str(cluster), linewidth=2, color=colors[cluster - 1])
        ax[i, 4].set_xlabel('% of days clear-dominated')
        ax[i, 4].set_ylabel('Height [km]')
        ax[i, 4].set_title('')
        ax[i, 4].set_xlim([0, 100])
        ax[i, 4].set_ylim([0, 10])
        ax[i, 4].set_title(season)

fig.tight_layout()
fig.savefig('mean_stats_obs_days_liquid.png', bbox_inches='tight')
microbase_avgs.close()
nsa_cluster.close()
