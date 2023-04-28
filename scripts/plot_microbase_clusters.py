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
fig, ax = plt.subplots(4, 4, figsize=(22, 20))
colors = ['b', 'g', 'k', 'c']
i = 0
def cluster_mean(x, cluster_no):
    return x.where(x.cluster == cluster_no).mean("time", skipna=True)

for cluster in [1, 2, 3, 4]:
    cmean = lambda x: cluster_mean(x, cluster)
    microbase_groupby = microbase_avgs.groupby("season").apply(cmean)
     
    print(microbase_groupby)
    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        microbase_groupby['Avg_Retrieved_LWC'].sel(season=season).T.plot(
            y="nheights", ax=ax[i, 0], label=str(cluster), color=colors[cluster - 1])

        ax[i, 0].set_xlabel('LWC [g $m^{-3}$]')
        ax[i, 0].set_ylabel('Height [km]')
        ax[i ,0].set_title('')
        ax[i, 0].set_xlim([0, 0.1])
        ax[i, 0].set_ylim([0, 10])
        ax[i, 0].set_title(season)
        ax[i, 0].legend()
    
        microbase_groupby['Avg_Retrieved_IWC'].sel(season=season).T.plot(
            y="nheights", ax=ax[i, 1], label=str(cluster), color=colors[cluster - 1])
        ax[i, 1].set_xlabel('IWC [g $m^{-3}$]')
        ax[i, 1].set_ylabel('Height [km]')
        ax[i, 1].set_title('')
        ax[i, 1].set_ylim([0, 10])
        ax[i, 1].set_xlim([0, 0.03])
        ax[i, 1].set_title(season)

        (1 - microbase_groupby['pct_clear']).sel(season=season).T.plot(
            y="nheights", ax=ax[i, 2], label=str(cluster), color=colors[cluster - 1])
        ax[i, 2].set_xlabel('Cloud fraction')
        ax[i, 2].set_ylabel('Height [km]')
        ax[i, 2].set_title('')
        ax[i, 2].set_xlim([0, 1])
        ax[i, 2].set_ylim([0, 10])
        ax[i, 2].set_title(season)
        
        microbase_groupby['mpc_occurrence'].sel(season=season).T.plot(
            y="nheights", ax=ax[i, 3], label=str(cluster), color=colors[cluster - 1])
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
