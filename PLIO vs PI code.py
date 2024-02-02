#%% ##--- MODULES ---##
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy.ndimage import maximum_filter, minimum_filter


#%% ##--- OPEN DATASET & ADD VARIABLES ---##
yr_2100_2129 = 'C:/Users/ikeva/OneDrive/Documenten/AW Studie/Thesis/Data/b.PLIO_5Ma_Eoi400_f19g16_NESSC_control_extra_daily_vars.cam2.h1.2100_2129.nc'
ds0 = xr.open_dataset(yr_2100_2129)
PI_6500_6529 = 'C:/Users/ikeva/OneDrive/Documenten/AW Studie/Thesis/Data/PI 6500-6529/b.PI_1pic_f19g16_NESSC_control_restart_2500_palaeo_vdc_overflows_tidal_off_extra_daily_vars.cam2.h1.6500_6529.nc'
ds1 = xr.open_dataset(PI_6500_6529)

PREC_PLIO = ds0.PRECC.copy(deep=True); SLP_PLIO = ds0.PSL.copy(deep=True)
PREC_PLIO.values = ds0.PRECC.values + ds0.PRECL.values
SLP_PLIO.values = ds0.PSL.values / 100
PREC_PLIO = PREC_PLIO.assign_attrs(long_name="Total precipitation rate (convective + large-scale)")
SLP_PLIO = SLP_PLIO.assign_attrs(units="hPa")
ds0['PREC'] = PREC_PLIO; ds0['SLP'] = SLP_PLIO

PREC_PI = ds1.PRECC.copy(deep=True); SLP_PI = ds1.PSL.copy(deep=True)
PREC_PI.values = ds1.PRECC.values + ds1.PRECL.values
SLP_PI.values = ds1.PSL.values / 100
PREC_PI = PREC_PI.assign_attrs(long_name="Total precipitation rate (convective + large-scale)")
SLP_PI = SLP_PI.assign_attrs(units="hPa")
ds1['PREC'] = PREC_PI; ds1['SLP'] = SLP_PI

AV_TEMP_PI = ds1.TREFHT.copy(deep=True); AV_TEMP_PLIO = ds0.TREFHT.copy(deep=True)
arr1 = sum(ds1.TREFHT.values)/10950
arr0 = sum(ds0.TREFHT.values)/10950
for t in range(len(ds1.TREFHT.values)):
    AV_TEMP_PI.values[t] = arr1
    AV_TEMP_PLIO.values[t] = arr0
AV_TEMP_PI = AV_TEMP_PI.assign_attrs(long_name="30-year averaged reference height temperature")
AV_TEMP_PLIO = AV_TEMP_PLIO.assign_attrs(long_name="30-year averaged reference height temperature")
ds1['TAV'] = AV_TEMP_PI; ds0['TAV'] = AV_TEMP_PLIO

AV_PREC_PI = ds1.PREC.copy(deep=True); AV_PREC_PLIO = ds0.PREC.copy(deep=True)
arr1 = sum(ds1.PREC.values)/10950
arr0 = sum(ds0.PREC.values)/10950
for t in range(len(ds1.PREC.values)):
    AV_PREC_PI.values[t] = arr1
    AV_PREC_PLIO.values[t] = arr0
AV_PREC_PI = AV_PREC_PI.assign_attrs(long_name="30-year averaged total precipitation rate")
AV_PREC_PLIO = AV_PREC_PLIO.assign_attrs(long_name="30-year averaged total precipitation rate")
ds1['PAV'] = AV_PREC_PI; ds0['PAV'] = AV_PREC_PLIO

TREFHT_C_PI = ds1.TREFHT.copy(deep=True); TREFHT_C_PLIO = ds0.TREFHT.copy(deep=True)
TREFHT_C_PI.values = ds1.TREFHT.values - 273; TREFHT_C_PLIO.values = ds0.TREFHT.values - 273
TREFHT_C_PI = TREFHT_C_PI.assign_attrs(units="°C"); TREFHT_C_PLIO = TREFHT_C_PLIO.assign_attrs(units="°C")
ds1['TREFHT_C'] = TREFHT_C_PI; ds0['TREFHT_C'] = TREFHT_C_PLIO

PREC_MMDAY_PI = ds1.PREC.copy(deep=True); PREC_MMDAY_PLIO = ds0.PREC.copy(deep=True)
PREC_MMDAY_PI.values = ds1.PREC.values * (3600*24*1000); PREC_MMDAY_PLIO.values = ds0.PREC.values * (3600*24*1000)
PREC_MMDAY_PI = PREC_MMDAY_PI.assign_attrs(units="mm/day"); PREC_MMDAY_PLIO = PREC_MMDAY_PLIO.assign_attrs(units="mm/day")
ds1['PREC_MMDAY'] = PREC_MMDAY_PI; ds0['PREC_MMDAY'] = PREC_MMDAY_PLIO


#%% ##--- SOME FUNCTIONS ---##
def ds_name(dataset):
    if dataset == "ds0":
        ds = ds0
        period = "Pliocene"
        first_year = 2100
    elif dataset == "ds1":
        ds = ds1
        period = "Pre-Industrial"
        first_year = 6500
    return ds, period, first_year

def make_cyclic(ds, coord):
    data_continued, lon_continued = add_cyclic_point(ds.values, coord=ds[f'{coord}'])
    ds_continued = xr.DataArray(data_continued, coords=[ds.lat, lon_continued], dims=["lat", "lon"], attrs=ds.attrs)
    return ds_continued
    
def dates_ds(dataset):
    dates = []
    for d in range(len(dataset.date.values)):
        dateformat = str(dataset.date.values[d])
        dates.append(dateformat)
        dates[d] = dateformat[:4] + '-' + dateformat[4:6] + '-' + dateformat[6:]
    dateset = xr.DataArray(dates, coords=[dataset.date], dims=["time"], attrs=dataset.time.attrs)
    return dateset  
          
def plot_maxmin_points(ax, lon, lat, data, extreme, nsize, symbol, color, transform=ccrs.PlateCarree()):
    if (extreme == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extreme == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')  
    mxy, mxx = np.where(data_ext==data)
    for i in range(1,len(mxy)):
        if mxy[i-1] > mxy[i] + 10 or mxx[i-1] > mxx[i] + 10:
            t = ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color, fontname='serif', size=20,
                    clip_on=True, horizontalalignment='center', verticalalignment='center', transform=transform)
            t.clipbox = ax.bbox  
            
def NAO_to_list(ds, yr_start, yrs, month):
    NAOv, SD = NAO_index_in_period(ds, year_start=yr_start, years=yrs, months=month, plot=False)
    lv = [list(NAOv[k]) for k in NAOv.keys()]
    NAO_lst = np.array(sum(lv, []))
    return NAO_lst
            
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)  


#%% ##--- SINGLE PLOT FOR VARIABLE ---##
            
def world_plot_variable(dataset1, dataset2, variable, time, zoom_ATL=False, zoom_EU=False, slp_contours=False, diffs=False):
    if diffs == True:
        ds1, period1, first_year1 = ds_name(dataset1)
        ds2, period2, first_year2 = ds_name(dataset2)
        ds_variable1 = ds1[f'{variable}'].isel(time=time)
        ds_variable2 = ds2[f'{variable}'].isel(time=time)
        ds_variable_diff = ds_variable2 - ds_variable1
        ds_cyclic = make_cyclic(ds=ds_variable_diff, coord='lon')
    else:
        ds, period, first_year = ds_name(dataset)
        ds_variable = ds[f'{variable}'].isel(time=time)
        ds_cyclic = make_cyclic(ds=ds_variable, coord='lon')
    
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(color='black', linestyle=':', draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    if diffs == True:
        if variable == 'PAV':
            ds_cyclic.plot.contourf(cmap=plt.cm.GnBu, vmin=-1*(10**-8), vmax=10*(10**-8), levels=23, transform=ccrs.PlateCarree(), cbar_kwargs={"label": "Precipitation rate (m/s)"}) 
        if variable == 'TAV':
            ds_cyclic.plot.contourf(cmap=plt.cm.Reds, vmin=0, vmax=30, levels=16, transform=ccrs.PlateCarree(), cbar_kwargs={"label": "Temperature (K)"})
    else:
        if variable == 'PREC' or variable == 'PAV':
            ds_cyclic.plot.contourf(cmap=plt.cm.GnBu, vmin=0, vmax=1.8*(10**-7), levels=10, transform=ccrs.PlateCarree(), cbar_kwargs={"label": "Precipitation rate (m/s)"}) 
        elif variable == 'TREFHT':
            ds_cyclic.plot.contourf(cmap=plt.cm.RdYlBu_r, vmin=210, vmax=315, levels=15, transform=ccrs.PlateCarree(), cbar_kwargs={"label": "Temperature (K)"})
        elif variable == 'TREFHT_C':
            ds_cyclic.plot.contourf(cmap=plt.cm.RdYlBu_r, vmin=-60, vmax=45, levels=15, transform=ccrs.PlateCarree(), cbar_kwargs={"label": "Temperature (°C)"})
   
    if zoom_ATL == True:
        ax.set_extent([-90, 60, 30, 90], crs=ccrs.PlateCarree())
    if zoom_EU == True:
        ax.set_extent([-25, 40, 30, 75], crs=ccrs.PlateCarree())
    
    if slp_contours == True:
        slp_cyclic = make_cyclic(ds=ds.SLP.isel(time=time), coord='lon')
        contours = slp_cyclic.plot.contour(colors='k', levels=15, linewidths=0.7, transform=ccrs.PlateCarree())
        contours.clabel(inline=True, inline_spacing=2, fontsize=8, fmt='%d', colors='black')
                
        lons, lats = np.meshgrid(ds0.lon.values, ds0.lat.values)
        mslp = ds.SLP.isel(time=time).values
        plot_maxmin_points(ax, lons, lats, mslp, 'max', 50, symbol='H', color='black', transform=ccrs.PlateCarree())
        plot_maxmin_points(ax, lons, lats, mslp, 'min', 25, symbol='L', color='black', transform=ccrs.PlateCarree())
    
    if diffs == True:
        plt.title(ds1[f'{variable}'].long_name, loc='left', fontweight='heavy')
        plt.title(f"{period2} - {period1}", loc='right')
    else:
        if variable == 'TAV' or variable == 'PAV':
            plt.title(ds[f'{variable}'].long_name, loc='left', fontweight='heavy')
            plt.title(f"{period}", loc='right')
        else:
            plt.title(f"{ds_variable.time.values}, {period}")
    plt.show()
    return None

print(world_plot_variable("ds1", "ds0", 'TREFHT_C', time=0, zoom_ATL=False, zoom_EU=False, slp_contours=False, diffs=False))


#%% ##--- NAM/NAO INDEX ---##

def Calculate_NAO_index(slp, lats=[35,65], lons=[290,30], norm=False):
    dl = 2
    p35 = slp.roll(lon=72, roll_coords=True).sel(lon=slice(lons[0],lons[1])).mean(dim='lon').sel(lat = slice(lats[0]-dl, lats[0]+dl)).mean(dim='lat')
    p65 = slp.roll(lon=72, roll_coords=True).sel(lon=slice(lons[0],lons[1])).mean(dim='lon').sel(lat = slice(lats[1]-dl, lats[1]+dl)).mean(dim='lat')

    pdiff = p35 - p65
    NAO = pdiff - pdiff.mean()
    
    if norm == True:
        NAO = NAO/NAO.std();        
    return NAO

def NAO_index_in_period(ds, year_start, years, months, plot=False):
    months_dict = {'Jan': [0,30], 'DJF': [333,423], 'July': [180,211], 'JJA': [150,242], 'Yearly': [0,365]}
    ticks_dict = {'Jan': mdates.DayLocator(interval=5), 'DJF': mdates.MonthLocator(interval=2), 'July': mdates.DayLocator(interval=5), 'JJA': mdates.MonthLocator(interval=1), 'Yearly': mdates.MonthLocator(interval=4)}
    rav_dict = {'Jan': 5, 'DJF': 10, 'July': 5, 'JJA': 10, 'Yearly': 30}
    month = months_dict[months]; ticks = ticks_dict[months]; rav = rav_dict[months]
    dateset = dates_ds(ds); first_year = int(dates_ds(ds).values[0][:4])
    
    if first_year == 6500:
        period = "Pre-Industrial"
    elif first_year == 2100:
        period = "Pliocene"
    
    slp_anomalies = ds.SLP.groupby("time.dayofyear") - ds.SLP.groupby("time.dayofyear").mean("time")
    NAO_values = {}
    
    if years < 2:
        slp_anomalies_in_period = slp_anomalies.isel(time=slice(month[0]+(year_start*365), month[1]+(year_start*365)))
        NAO_months = Calculate_NAO_index(slp_anomalies_in_period) 
        mean = NAO_months.rolling(time=rav).mean().dropna("time")
        NAO_months['time'] = dateset.isel(time=slice(month[0]+(year_start*365), month[1]+(year_start*365)))
        NAO_values.update({f"{months}" " " f"{year_start + first_year}": NAO_months.values})
        stds = np.std(NAO_values[f"{months}" " " f"{year_start + first_year}"])
        
        if plot == True:
            plt.figure(figsize=(20,5))
            NAO_months.plot(color='black', linewidth=0.75)
            plt.plot(mean, color='red', linewidth=2)
            plt.fill_between(NAO_months.time.values, NAO_months.values, 0, where=NAO_months.values>0, color='#ecb474', interpolate=True)
            plt.fill_between(NAO_months.time.values, NAO_months.values, 0, where=NAO_months.values<0, color='#95B3D7', interpolate=True)
            ax = plt.gca() 
            ax.xaxis.set_major_locator(ticks)
            for label in ax.get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            plt.ylabel("NAO index [%s]" %months)
            plt.show()
    
    elif years >= 2:
        NAO_idxs, dts, means, means_rav, stds = [], [], [], [], []
        num = 365*year_start
        stds_m, stds_mav =[], []

        for yr in range(year_start, year_start+years):
            slp_anomalies_in_period = slp_anomalies.isel(time=slice(month[0]+num, month[1]+num))            
            NAO_months = Calculate_NAO_index(slp_anomalies_in_period)
            mean = NAO_months.rolling(time=rav).mean().dropna("time")
            means.append(mean.values) ## NAO indices graph
            means_rav.extend(mean.values) ## running average graph
            NAO_months['time'] = dateset.isel(time=slice(month[0]+num, month[1]+num))
            
            date = NAO_months.time.values
            NAO_idx = NAO_months.values
            std = np.std(NAO_idx)
            dts.append(date)
            NAO_idxs.append(NAO_idx)
            stds.append(std)
            NAO_values.update({f"{months}" " " f"{yr + first_year}": NAO_idx})
           
            num += 365
            
            ## 30-year averaged SD per month
        #     for m in range(0,335,30):
        #         stds_m.append(np.std(NAO_idx[m:m+30]))
        # for n in range(len(stds_m)):
        #     if n+12<=23:
        #         stds_mav.append(sum(stds_m[n::12])/len(stds_m[n::12]))
        # print(stds_mav)


        if plot == True:
            ## NAO idx
            fig, ax = plt.subplots(nrows=1, ncols=years, sharey=True, figsize=(20,5))
            for p in range(years):                               
                ax[p].plot(dts[p], NAO_idxs[p], color='black', linewidth=0.75)
                ax[p].plot(means[p], color='red', linewidth=2)
                ax[p].xaxis.set_major_locator(ticks)
                for label in ax[p].get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')
                ax[p].spines['right'].set_visible(False)
                if p < years-1:
                    ax[p+1].spines['left'].set_visible(False)
                ax[p].fill_between(dts[p], NAO_idxs[p], 0, where=NAO_idxs[p]>0, color='#ecb474', interpolate=True)
                ax[p].fill_between(dts[p], NAO_idxs[p], 0, where=NAO_idxs[p]<0, color='#95B3D7', interpolate=True)
            ax[0].set_ylabel("NAO index (hPa)", fontsize=18)
            ax[0].set_yticks(np.arange(-30,31,10))
            plt.suptitle(f"{period} [{months}]", fontweight='bold', fontsize='x-large')
            plt.subplots_adjust(wspace=0.05)
            plt.show()
            
            ## running average
            plt.figure(figsize=(15,5))
            plt.plot(means_rav, 'r')
            x = np.arange(0,len(means),1)
            plt.plot(x, np.zeros(len(x)), 'black', linestyle='--', linewidth=0.7)
            y = np.arange(-15, 16, 10)
            plt.yticks(ticks=y)
            plt.xlabel("Time (years)", fontsize=16)
            plt.ylabel("NAO index (hPa)", fontsize=16)
            plt.title(f"Running average smoothed over {rav} days", fontsize=16)
            plt.suptitle(f"{period} [{months}]", fontweight='bold', fontsize='x-large')
            plt.show()
    
    return NAO_values, stds

print(NAO_index_in_period(ds1, year_start=0, years=3, months='Yearly', plot=True))


#%% ##--- HISTOGRAM ---##
month = 'Jan'
yrs = 30
NAO_lst_PI = NAO_to_list(ds1, 0, yrs, month)
NAO_lst_PLIO = NAO_to_list(ds0, 0, yrs, month)

pressures= np.arange(-30,31,2)
plt.style.use('seaborn-v0_8-deep')
plt.hist(NAO_lst_PI, bins=pressures, alpha=0.9, edgecolor='black', linewidth=0.5, orientation='horizontal', label='Pre-Industrial')
plt.hist(NAO_lst_PLIO, bins=pressures, alpha=0.6, edgecolor='black', linewidth=0.5, orientation='horizontal', label='Pliocene')

plt.title(f"{month} (30-yrs) pressure anomaly distribution", fontsize=12)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Pressure anomaly (hPa)", fontsize=12)
plt.legend()
plt.show()



#%% ##--- STANDARD DEVIATIONS CALCULATION ---##
NAO_jan_PI, SD_jan_PI = NAO_index_in_period(ds1, year_start=0, years=30, months='Jan', plot=False)
NAO_djf_PI, SD_djf_PI = NAO_index_in_period(ds1, year_start=0, years=30, months='DJF', plot=False)
NAO_july_PI, SD_july_PI = NAO_index_in_period(ds1, year_start=0, years=30, months='July', plot=False)
NAO_jja_PI, SD_jja_PI = NAO_index_in_period(ds1, year_start=0, years=30, months='JJA', plot=False)
NAO_y_PI, SD_y_PI = NAO_index_in_period(ds1, year_start=0, years=30, months='Yearly', plot=False)

NAO_jan_PLIO, SD_jan_PLIO = NAO_index_in_period(ds0, year_start=0, years=30, months='Jan', plot=False)
NAO_djf_PLIO, SD_djf_PLIO = NAO_index_in_period(ds0, year_start=0, years=30, months='DJF', plot=False)
NAO_july_PLIO, SD_july_PLIO = NAO_index_in_period(ds0, year_start=0, years=30, months='July', plot=False)
NAO_jja_PLIO, SD_jja_PLIO = NAO_index_in_period(ds0, year_start=0, years=30, months='JJA', plot=False)
NAO_y_PLIO, SD_y_PLIO = NAO_index_in_period(ds0, year_start=0, years=30, months='Yearly', plot=False)

sd_PI = [SD_jan_PI, SD_djf_PI, SD_july_PI, SD_jja_PI, SD_y_PI]
sd_PLIO = [SD_jan_PLIO, SD_djf_PLIO, SD_july_PLIO, SD_jja_PLIO, SD_y_PLIO]
dev_PI = [sd_PI[i]-np.mean(sd_PI[i]) for i in range(5)]
dev_PLIO = [sd_PLIO[i]-np.mean(sd_PLIO[i]) for i in range(5)]


#%% ##--- STANDARD DEVIATIONS PLOT ---##
periods = ["Pre-Industrial", "Pliocene"]   
m = ['January', 'DJF', 'July', 'JJA', 'Year']
c = ['olivedrab', 'cadetblue', 'orange', 'firebrick', 'indigo']
d = ['deviations_jan', 'deviations_djf', 'deviations_july', 'deviations_jja', 'deviations_year']
marker = ['o', '+', 'x', '^', 's']
t = np.arange(0, 30)    

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,5))
for i in range(5):
    axs[1].scatter(t, sd_PLIO[i], marker=marker[i], label=f"{m[i]}")
    axs[0].scatter(t, sd_PI[i], marker=marker[i], label=f"{m[i]}")
for a,ax in enumerate(axs):
    ax.set_title(f"Standard deviation per year ({periods[a]})")
    ax.set_xlabel("Time (years)")
    ax.legend(loc='upper right')
axs[0].set_ylabel("Deviation (hPa)")
plt.subplots_adjust(wspace=0.05)
plt.show()


#%% ##--- 10 PERCENT HIGHEST/LOWEST ---##

def highest_lowest_percentile(dataset, variable, month, NAO, percent, year=None):
    slice_dict = {'Jan': [30, 0], 'DJF': [90, 333], 'July': [31, 180], 'JJA': [92, 150]}
    tslice = slice_dict[month][0]; tdur = slice_dict[month][1]
    
    if percent == 'lowest':
        perc = NAO[NAO < np.percentile(NAO,10)]
    elif percent == 'highest':
        perc = NAO[NAO > np.percentile(NAO,90)]
        
    tidx = []
    for a in perc:
        idx = np.where(NAO == a)[0][0]
        if year != None:
            tmax = idx + tdur + 365*year
        elif year == None:
            tmax = idx + tdur
        tsel = tmax - (tslice*(idx//tslice)) + (365*(idx//tslice))
        tidx.append(tsel)

    perc_dsn = dataset[variable].copy(deep=True); slp_dsn = dataset['SLP'].copy(deep=True)
    sets_var, sets_slp = [], []
    
    for x in tidx:
        var_sel = perc_dsn.sel(time=np.isin(perc_dsn.time, perc_dsn['time'][x]))
        slp_sel = slp_dsn.sel(time=np.isin(slp_dsn.time, slp_dsn['time'][x]))
        sets_var.append(var_sel), sets_slp.append(slp_sel)

    perc_ds = xr.concat(sets_var, dim='time'); slp_ds = xr.concat(sets_slp, dim='time')
        
    if year != None:
        return perc_ds, slp_ds, tidx
    else:
        return perc_ds, slp_ds       
    
    
#%% ##--- STORM TRACKS ---##
def lowest_slp_atl(ds, start, percentiles, percent, NAO): 
    time_list, lowest_slp = [], []; lat_dict, lon_dict = {}, {}
    n, m = 1, 1
    
    if percentiles == False:
        for j in ds.SLP.time.values[333+start:423+start]: 
            time_list.append(j)
    elif percentiles == True:
        perc_ds, slp_ds = highest_lowest_percentile(ds, variable='SLP', month='DJF', NAO=NAO, percent=percent)
        for j in slp_ds.time.values:
            time_list.append(j)

    for k in range(len(time_list)):
        if percentiles == False:
            dsroll_cyclic = make_cyclic(ds.SLP.sel(time=time_list[k]), 'lon')
        elif percentiles == True:
            dsroll_cyclic = make_cyclic(slp_ds.sel(time=time_list[k]), 'lon')
        dsroll = dsroll_cyclic.roll(lon=72, roll_coords=True)
        slp = dsroll.sel(lat=slice("30", "80"), lon=slice(290, 10))

        v_slp = slp.values
        slp_min = v_slp.min()
        lowest_slp.append(slp_min)
        if slp_min < 1010:
            pos = slp.argmin(...)
            latpos, lonpos = pos['lat'].item(), pos['lon'].item()        
            if k == 0:
                lat_dict.update({"1": [slp['lat'][latpos].item()]})
                lon_dict.update({"1": [slp['lon'][lonpos].item()]})
            if k >= 1:
                lat_diff = slp['lat'][latpos].item() - lat_dict[f"{n}"][m-1]
                lon_diff = slp['lon'][lonpos].item() - lon_dict[f"{n}"][m-1]
                if np.sqrt((lat_diff ** 2) + (lon_diff ** 2)) < 50 and lon_diff >= 0:
                    lat_dict[f"{n}"] += [slp['lat'][latpos].item()]
                    lon_dict[f"{n}"] += [slp['lon'][lonpos].item()]
                    m += 1
                else:
                    n += 1
                    lat_dict.update({f"{n}": [slp['lat'][latpos].item()]})
                    lon_dict.update({f"{n}": [slp['lon'][lonpos].item()]})
                    m = len(lat_dict[f"{str(n)}"])
        else:
            continue
    return lowest_slp, lat_dict, lon_dict

def longest_storm_tracks(dataset, yrs, percentiles, percent):
    ds, period, first_year = ds_name(dataset)  
    all_lats, all_lons = [], []
    
    if percentiles == False:
        for yr in range(yrs):
            slp_lows, lat_dict, lon_dict = lowest_slp_atl(ds, start=365*yr, percentiles=False, percent=None, NAO=None)   
            combidict = {}      
            lenscombi = np.zeros(len(lat_dict), dtype=int)
            for c in range(1,len(lat_dict)):
                combidict.update({f"{c}": []})
                combidict[f"{c}"] += [lat_dict[f"{c}"], lon_dict[f"{c}"]]
                lenscombi[c-1] = len(combidict[f"{c}"][0]) + len(combidict[f"{c}"][1])               
            perc = lenscombi[lenscombi > np.percentile(lenscombi,90)]  
            idxs=[]
            for pr in perc:
                idx = np.where(lenscombi == pr)[0]
                for i in idx:
                    if i not in idxs:
                        idxs.append(i)
            lats, lons = [], []
            for idxp in idxs:
                pos = idxp+1
                lats.append(combidict[f"{pos}"][0])
                lons.append(combidict[f"{pos}"][1])
            all_lats.extend(lats)
            all_lons.extend(lons)
        
    elif percentiles == True:
        NAO_lst = NAO_to_list(ds, 0, 30, 'DJF')
        slp_lows, lat_dict, lon_dict = lowest_slp_atl(ds, start=0, percentiles=percentiles, percent=percent, NAO=NAO_lst)
        for latv in lat_dict.values():
            all_lats.append(latv)
        for lonv in lon_dict.values():
            all_lons.append(lonv)
             
    return all_lats, all_lons, slp_lows
    

#%%      
all_lats_PI, all_lons_PI, slp_lows_PI = longest_storm_tracks("ds1", 29, percentiles=False, percent=None)
all_lats_PLIO, all_lons_PLIO, slp_lows_PLIO = longest_storm_tracks("ds0", 29, percentiles=False, percent=None)

all_lats_PI_l, all_lons_PI_l, slp_lows_PI_l = longest_storm_tracks("ds1", 29, percentiles=True, percent='lowest')
all_lats_PLIO_l, all_lons_PLIO_l, slp_lows_PLIO_l = longest_storm_tracks("ds0", 29, percentiles=True, percent='lowest')

all_lats_PI_h, all_lons_PI_h, slp_lows_PI_h = longest_storm_tracks("ds1", 29, percentiles=True, percent='highest')
all_lats_PLIO_h, all_lons_PLIO_h, slp_lows_PLIO_h = longest_storm_tracks("ds0", 29, percentiles=True, percent='highest') 


#%% ##--- TIMESERIES CORRELATION---##
def Atlantic_timeseries(dataset, variable, month, percentiles, percent, slp_contours):
    region = [270, 60, 30, 90]
    lats = [lt for lt in ds0.lat.sel(lat=slice(region[2],region[3])).values]
    lons = [ln for ln in ds0.lon.roll(lon=72, roll_coords=True).sel(lon=slice(region[0],region[1])).values]
    ds, period, first_year = ds_name(dataset)
    NAO_lst = NAO_to_list(ds, 0, 30, month)

    month_idxs = ds.groupby('time.month').groups
    months_idxs_dict = {'Jan': month_idxs[1], 'DJF': month_idxs[1][30:] + month_idxs[2][29:] + month_idxs[12][:-31], 'July': month_idxs[7], 'JJA': month_idxs[6] + month_idxs[7] + month_idxs[8]}
    if month == 'Jan':
        idxs_jan = months_idxs_dict[month][:30]
        composite_list = [months_idxs_dict[month][x:x+30] for x in range(31, len(months_idxs_dict[month]),31)]
        idxs_jan.extend(np.ravel(composite_list))
        ds_month = ds.isel(time=idxs_jan)
    elif month == 'DJF' or month == 'JJA':
        ds_month = ds.isel(time=months_idxs_dict[month])
        ds_month = ds_month.sortby('time')
    else:
        ds_month = ds.isel(time=months_idxs_dict[month])   
    
    if percentiles == True:
        percentiles_ds, slp_ds = highest_lowest_percentile(ds, variable=variable, month=month, NAO=NAO_lst, percent=percent)

    tsm = np.zeros((len(lats), len(lons)), dtype=np.ndarray)
    mean_ts = np.zeros((len(lats), len(lons)))
    mean_ts_perc = np.zeros((len(lats), len(lons)))
    mean_ts_to_anomaly = np.zeros((len(lats), len(lons)))
    mean_slp_ts = np.zeros((len(lats), len(lons)))
    sd_ts = np.zeros((len(lats), len(lons)))
    NAO_ts = np.zeros((len(lats), len(lons)), dtype=np.ndarray)
    corr_ts = np.zeros((len(lats), len(lons)))

    for i in range(len(lats)):
        for j in range(len(lons)):            
            if percentiles == True:
                if variable == 'TREFHT_C':
                    mean_ts_perc[i,j] = percentiles_ds.sel(lon=lons[j], lat=lats[i]).mean(dim='time').values
                    mean_ts_to_anomaly[i,j] = ds_month[variable].sel(lon=lons[j], lat=lats[i]).mean(dim='time').values
                    mean_ts[i,j] = mean_ts_perc[i,j] - mean_ts_to_anomaly[i,j]
                elif variable == 'PREC_MMDAY':
                    mean_ts[i,j] = percentiles_ds.sel(lon=lons[j], lat=lats[i]).mean(dim='time').values
                if slp_contours == True:
                    mean_slp_ts[i,j] = slp_ds.sel(lon=lons[j], lat=lats[i]).mean(dim='time').values               
            else:
                if month == 'DJF':
                    NAO_lst = NAO_lst[:2610]
                    
                tsm[i,j] = ds_month[variable].sel(lon=lons[j], lat=lats[i]).values
                mean_ts[i,j] = ds_month[variable].sel(lon=lons[j], lat=lats[i]).mean(dim='time').values
                sd_ts[i,j] = ds_month[variable].sel(lon=lons[j], lat=lats[i]).std(dim='time').values
                NAO_ts[i,j] = NAO_lst
                corr = np.corrcoef(tsm[i][j], NAO_ts[i][j])
                corr_ts[i,j] = corr[0][1]
                mean_slp_ts[i,j] = ds_month['SLP'].sel(lon=lons[j], lat=lats[i]).mean(dim='time').values  

    return mean_ts, sd_ts, corr_ts, mean_slp_ts

def Atlantic_data(dataset, variable, values, analysis, month, slp_contours, mslp):
    ds, period, first_year = ds_name(dataset)
    
    nds = ds[variable].isel(time=0).copy(deep=True)
    nds = nds.where((nds['lat'] < -90) | (nds['lat'] > 30), drop=True)
    nds = nds.where((nds['lon'] < 62.5) | (nds['lon'] > 267.5), drop=True)
    nds = nds.roll(lon=36, roll_coords=True)
    
    nds.values = values
    new_lons = nds.lon.values.copy()
    new_lons[1:] += np.cumsum(np.diff(nds.lon.values) < -180) * 360
    nds['lon'] = new_lons
          
    if variable == 'SLP' and analysis == 'mean':
        cmap = plt.cm.viridis
    elif variable == 'SLP' and analysis == 'variation (SD)':
        cmap = plt.cm.coolwarm
    elif variable == 'TREFHT' or variable == 'TREFHT_C':
        cmap = plt.cm.RdYlBu_r
    elif variable == 'PREC' or variable == 'PREC_MMDAY':
        cmap = plt.cm.GnBu
        
    if slp_contours == True:
        nds_slp = nds.copy(deep=True)
        nds_slp.values = mslp    
        return nds, nds_slp, cmap
    elif slp_contours == False:
        return nds, cmap

variable = 'PREC_MMDAY'
month = 'DJF'
percent = 'highest'
analysis = ['mean', 'variation (SD)', f'mean of 10% {percent} NAO days', f'10% {percent} anomaly', 'Storm tracks', 'Storm tracks 10% NAO-']
xpos = [0.23, 0.28, 0.35, 0.31, 0.43, 0.47]
cbar_dict = {'meanSLP': [993,1035,15], 'sdSLP': [3,18,14], 'sdSLP1': [1,12,11], 'mTREFHT': [-45,25,15], 'mTREFHT1': [-20,50,15], 'mPREC': [0,12,13], 'mPREC1': [0,8,17], 'Tanomaly': [-8,8,17], 'Tanomaly1': [-4,4,17]}
an = analysis[4]; X = xpos[4]; vmin = cbar_dict['mPREC1'][0]; vmax = cbar_dict['mPREC1'][1]; level = cbar_dict['mPREC1'][2]
percentiles = False; slp_contours = False; storm_tracks = True; corr_contours = False

mean_ts_PI, sd_ts_PI, corr_ts_PI, mean_slp_ts_PI = Atlantic_timeseries("ds1", variable, month, percentiles, percent, slp_contours)
mean_ts_PLIO, sd_ts_PLIO, corr_ts_PLIO, mean_slp_ts_PLIO = Atlantic_timeseries("ds0", variable, month, percentiles, percent, slp_contours)

using_PI = mean_ts_PI
using_PLIO = mean_ts_PLIO

if slp_contours == True:
    nds_PI, nds_slp_PI, cmap_PI = Atlantic_data("ds1", variable, using_PI, an, month, slp_contours, mslp=mean_slp_ts_PI)
    nds_PLIO, nds_slp_PLIO, cmap_PLIO = Atlantic_data("ds0", variable, using_PLIO, an, month, slp_contours, mslp=mean_slp_ts_PLIO) 
elif slp_contours == False:
    nds_PI, cmap_PI = Atlantic_data("ds1", variable, using_PI, an, month, slp_contours, mslp=None)
    nds_PLIO, cmap_PLIO = Atlantic_data("ds0", variable, using_PLIO, an, month, slp_contours, mslp=None)    
    if corr_contours == True:
        corr_nds_PI, corr_cmap_PI = Atlantic_data("ds1", variable, corr_ts_PI, an, month, slp_contours, mslp=None)
        corr_nds_PLIO, corr_cmap_PLIO = Atlantic_data("ds0", variable, corr_ts_PLIO, an, month, slp_contours, mslp=None)

fig, axs = plt.subplots(nrows=2, figsize=(15,10), subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
for ax in axs:
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(color='black', linestyle=':', draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}
    ax.set_extent([-90, 60, 31, 90], crs=ccrs.PlateCarree())
    if storm_tracks == True:
        ax.set_extent([-70, 10, 40, 85], crs=ccrs.PlateCarree())

nds_cyclic_PI = make_cyclic(ds=nds_PI, coord='lon')
nds_cyclic_PLIO = make_cyclic(ds=nds_PLIO, coord='lon')

im1 = nds_cyclic_PI.plot.contourf(ax=axs[0], cmap=cmap_PI, vmin=vmin, vmax=vmax, levels=level, transform=ccrs.PlateCarree(), add_colorbar=False)
im2 = nds_cyclic_PLIO.plot.contourf(ax=axs[1], cmap=cmap_PLIO, vmin=vmin, vmax=vmax, levels=level, transform=ccrs.PlateCarree(), add_colorbar=False)
cb = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation="vertical", extend='both')
if variable == 'TREFHT_C':
    cb.set_label(label='Temperature (°C)', size=16, labelpad=12)
    plt.suptitle(f"SAT {an} in {month}", fontsize='xx-large', fontweight='semibold', x=X, y=0.91)
elif variable == 'PREC_MMDAY':
    cb.set_label(label='Precipitation rate (mm/day)', size=16, labelpad=12)
    plt.suptitle(f"{an} in {month}", fontsize='xx-large', fontweight='semibold', x=X, y=0.91)

if storm_tracks == True and percentiles == False:
    for q in range(len(all_lats_PI)):
        axs[0].plot(all_lons_PI[q], all_lats_PI[q], color='maroon', linestyle=(0, (5, 5)), linewidth=1.2, transform=ccrs.PlateCarree())
    for r in range(len(all_lats_PLIO)):
        axs[1].plot(all_lons_PLIO[r], all_lats_PLIO[r], color='maroon', linestyle=(0, (5, 5)), linewidth=1.2, transform=ccrs.PlateCarree())

if storm_tracks == True and percentiles == True:
    if percent == 'lowest':
        for q in range(len(all_lats_PI_l)):
            axs[0].plot(all_lons_PI_l[q], all_lats_PI_l[q], color='maroon', linestyle=(0, (5, 5)), linewidth=1.2, transform=ccrs.PlateCarree())
        for r in range(len(all_lats_PLIO_l)):
            axs[1].plot(all_lons_PLIO_l[r], all_lats_PLIO_l[r], color='maroon', linestyle=(0, (5, 5)), linewidth=1.2, transform=ccrs.PlateCarree())
    elif percent == 'highest':
        for q in range(len(all_lats_PI_h)):
            axs[0].plot(all_lons_PI_h[q], all_lats_PI_h[q], color='maroon', linestyle=(0, (5, 5)), linewidth=1.2, transform=ccrs.PlateCarree())
        for r in range(len(all_lats_PLIO_h)):
            axs[1].plot(all_lons_PLIO_h[r], all_lats_PLIO_h[r], color='maroon', linestyle=(0, (5, 5)), linewidth=1.2, transform=ccrs.PlateCarree())


if percentiles == False and slp_contours == False and corr_contours == True:
    contours_corr_cyclic_PI = make_cyclic(ds=corr_nds_PI, coord='lon')
    contours_corr_PI = contours_corr_cyclic_PI.plot.contour(ax=axs[0], colors='white', levels=15, linewidths=0.7, transform=ccrs.PlateCarree())
    contours_corr_PI.clabel(inline=True, inline_spacing=2, fontsize=14, fmt='%.1f', colors='white')
    contours_corr_cyclic_PLIO = make_cyclic(ds=corr_nds_PLIO, coord='lon')
    contours_corr_PLIO = contours_corr_cyclic_PLIO.plot.contour(ax=axs[1], colors='white', levels=15, linewidths=0.7, transform=ccrs.PlateCarree())
    contours_corr_PLIO.clabel(inline=True, inline_spacing=2, fontsize=14, fmt='%.1f', colors='white')

if slp_contours == True:
    slp_PI_cyclic = make_cyclic(nds_slp_PI, 'lon')
    slp_PLIO_cyclic = make_cyclic(nds_slp_PLIO, 'lon')
    contours_PI = slp_PI_cyclic.plot.contour(ax=axs[0], colors='k', levels=15, linewidths=0.7, transform=ccrs.PlateCarree())
    contours_PLIO = slp_PLIO_cyclic.plot.contour(ax=axs[1], colors='k', levels=15, linewidths=0.7, transform=ccrs.PlateCarree())
    contours_PI.clabel(inline=True, inline_spacing=2, fontsize=11, fmt='%d', colors='black')
    contours_PLIO.clabel(inline=True, inline_spacing=2, fontsize=11, fmt='%d', colors='black')
    
    region = [270, 60, 30, 90]
    lat = [lt for lt in ds0.lat.sel(lat=slice(region[2],region[3])).values]
    lon = [ln for ln in ds0.lon.roll(lon=72, roll_coords=True).sel(lon=slice(region[0],region[1])).values]
    lons, lats = np.meshgrid(lon, lat)
    mslp_PI = nds_slp_PI.values
    mslp_PLIO = nds_slp_PLIO.values
    
    plot_maxmin_points(axs[0], lons, lats, mslp_PI, 'max', 15, symbol='H', color='black')
    plot_maxmin_points(axs[0], lons, lats, mslp_PI, 'min', 5, symbol='L', color='black')
    plot_maxmin_points(axs[1], lons, lats, mslp_PLIO, 'max', 15, symbol='H', color='black')
    plot_maxmin_points(axs[1], lons, lats, mslp_PLIO, 'min', 5, symbol='L', color='black')

axs[0].set_title("Pre-Industrial", fontsize='xx-large', loc='right')
axs[1].set_title("Pliocene", fontsize='xx-large', loc='right')
plt.show()


#%% ##--- TOTAL HIGH/LOW NAO DAYS ---##
def NAO_index_value(value_dict, first_year, year_start, years, months):
    NAO_highs, NAO_lows = {}, {}
    NAO_highs_num, NAO_lows_num = {}, {}
    
    for i in range(year_start, years):
        key = f"{months}" " " f"{i + first_year}"
        values = value_dict[key]
        NAO_highs.update({key: []}); NAO_lows.update({key: []})
        NAO_highs_num.update({key+" Highs": ""}); NAO_lows_num.update({key+" Lows": ""})
        
        for j in range(len(values)):
            if values[j] > values.mean() + np.std(values):
                NAO_highs[key] += [values[j]]
            elif values[j] < values.mean() - np.std(values):
                NAO_lows[key] += [values[j]]
                
        NAO_highs_num[key+" Highs"] += str(len(NAO_highs[key]))
        NAO_lows_num[key+" Lows"] += str(len(NAO_lows[key]))
                
    return NAO_highs_num, NAO_lows_num

dataset = "ds1"
ds, period, first_year = ds_name(dataset)
years = 10
year_start = 0
months = 'Jan'
NAO_values, SD = NAO_index_in_period(ds, year_start, years, months)

NAO_highs_num, NAO_lows_num = NAO_index_value(NAO_values, first_year, year_start, years, months)
# print(NAO_highs_num)
# print(NAO_lows_num)


#%% ##--- DAYS FOR NAO+/NAO- MAXIMA ---##
def NAO_max_min_dates(ds, first_year, year_start, years, months):
    NAO_values, SD = NAO_index_in_period(ds, year_start, years, months)  
    NAO_max_dates, NAO_min_dates = {}, {}
    maxs, mins = [], [] 
     
    for i in range(year_start, year_start+years):
        key = f"{months}" " " f"{i + first_year}"
        
        NAO_max = np.max(NAO_values[key])
        NAO_min = np.min(NAO_values[key])
        maxs.append(NAO_max), mins.append(NAO_min)

        NAO_max_t = np.argmax(NAO_values[key])
        NAO_min_t = np.argmin(NAO_values[key])
        NAO_max_dates[key] = NAO_max_t
        NAO_min_dates[key] = NAO_min_t

    # print(np.mean(maxs), np.mean(mins))
    # print(np.max(maxs), np.min(mins))
    # print(np.mean(meansmaxs), np.mean(meansmins))
  
    return NAO_values, NAO_max_dates, NAO_min_dates
print("PLIO")
print(NAO_max_min_dates(ds0, 2100, 0, 30, 'Jan'))
print("PI")
print(NAO_max_min_dates(ds1, 6500, 0, 30, 'Jan'))

#%% ##--- MEAN SAT/PR/MSLP N-EU, S-EU ---##

def variable_mean(dataset, year_start, years, months, variable, corr_plot=False):
    NEU = [47,70]
    SEU = [36,47]
    dateadds = {'Jan': 0, 'DJF': 333, 'July': 180, 'JJA': 150, 'Yearly': 0}
    NEU_maxs, SEU_maxs, NEU_mins, SEU_mins = [], [], [], []
    NEU_daily, SEU_daily = [], []
    
    ds, period, first_year = ds_name(dataset)
    NAO_values, max_dates, min_dates = NAO_max_min_dates(ds, first_year, year_start, years, months)
    
    for i in range(year_start, year_start+years):
        key = f"{months}" " " f"{i + first_year}"
        NAO_max_d = max_dates[key] + dateadds[f'{months}'] + i*365; NAO_min_d = min_dates[key] + dateadds[f'{months}'] + i*365
            
        N_mean_max = ds[f'{variable}'].isel(time=NAO_max_d).sel(lat=slice(NEU[0],NEU[1])).roll(lon=72, roll_coords=True).sel(lon=slice(350,40)).mean().values.item()
        S_mean_max = ds[f'{variable}'].isel(time=NAO_max_d).sel(lat=slice(SEU[0],SEU[1])).roll(lon=72, roll_coords=True).sel(lon=slice(350,40)).mean().values.item()        
        N_mean_min = ds[f'{variable}'].isel(time=NAO_min_d).sel(lat=slice(NEU[0],NEU[1])).roll(lon=72, roll_coords=True).sel(lon=slice(350,40)).mean().values.item()
        S_mean_min = ds[f'{variable}'].isel(time=NAO_min_d).sel(lat=slice(SEU[0],SEU[1])).roll(lon=72, roll_coords=True).sel(lon=slice(350,40)).mean().values.item()
        
        NEU_maxs.append(N_mean_max); SEU_maxs.append(S_mean_max)
        NEU_mins.append(N_mean_min); SEU_mins.append(S_mean_min)
        
        if corr_plot == True:
            for t in range(dateadds[f'{months}']+(i*365), dateadds[f'{months}']+(i*365)+len(NAO_values[key])):
                N_d = ds[f'{variable}'].isel(time=t).sel(lat=slice(NEU[0],NEU[1])).roll(lon=72, roll_coords=True).sel(lon=slice(350,40)).mean().values.item()
                S_d = ds[f'{variable}'].isel(time=t).sel(lat=slice(SEU[0],SEU[1])).roll(lon=72, roll_coords=True).sel(lon=slice(350,40)).mean().values.item()
                NEU_daily.append(N_d); SEU_daily.append(S_d)
    
    if corr_plot == True:
        NAOv = NAO_to_list(ds, year_start, years, months)
        fig, ax1 = plt.subplots()
        ax1.scatter(NAOv, NEU_daily, color='royalblue', label='Northern Europe')
        ax1.scatter(NAOv, SEU_daily, color='r', label='Southern Europe')
        
        z = np.polyfit(NAOv, NEU_daily, 1); zz = np.polyfit(NAOv, SEU_daily, 1)
        p = np.poly1d(z); pp = np.poly1d(zz)
        ax1.plot(NAOv, p(NAOv), color='darkblue')
        ax1.plot(NAOv, pp(NAOv), color='darkred')
        ax1.set_ylim([0,5])
        ax1.set_ylabel('Precipitation (mm/day)', fontsize=14)
        # ax1.set_ylim([5,35])
        # ax1.set_ylabel('Temperature (°C)', fontsize=14)
        # ax1.set_ylim([990,1030])
        # ax1.set_ylabel('Pressure (hPa)', fontsize=14)
        ax1.set_xlim([-25,25])
        ax1.set_xlabel('NAO index (hPa)', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.set_title(f'Link between NAO index and European SLP ({months})')
        plt.suptitle(f"{period}", fontsize='large', fontweight='semibold')
        plt.show()
        
        corr_NEU_NAO = np.corrcoef(NEU_daily, NAOv)
        corr_SEU_NAO = np.corrcoef(SEU_daily, NAOv)
        print(corr_NEU_NAO[0][1], corr_SEU_NAO[0][1])
    
    # return NEU_maxs, NEU_mins, SEU_maxs, SEU_mins
print(variable_mean(dataset="ds1", year_start=0, years=30, months='JJA', variable='PREC_MMDAY', corr_plot=True))
print(variable_mean(dataset="ds0", year_start=0, years=30, months='JJA', variable='PREC_MMDAY', corr_plot=True))

# m = 'DJF'
# v = 'TREFHT'
# print("PLIO")
# NEU_maxs1, NEU_mins1, SEU_maxs1, SEU_mins1 = variable_mean(dataset="ds0", year_start=0, years=30, months=m, variable=v)
# lst1 = [NEU_maxs1, NEU_mins1, SEU_maxs1, SEU_mins1]
# for item in lst1:
#     print(np.mean(item))
# print("")
# print("PI")
# NEU_maxs2, NEU_mins2, SEU_maxs2, SEU_mins2 = variable_mean(dataset="ds1", year_start=0, years=30, months=m, variable=v)
# lst2 = [NEU_maxs2, NEU_mins2, SEU_maxs2, SEU_mins2]
# for item in lst2:
#     print(np.mean(item))


#%% ##--- OPTIONAL -> COMPARISON PLOT PRECC/TREFHT NAO+/NAO- ---## 
# def subplots_NAO(dataset, variable, year, months, anomalies, extrema, zoom_EU):
#     ds, period, first_year = ds_name(dataset)
#     cmap_dict = {'TREFHT':plt.cm.RdYlBu_r, 'PREC':plt.cm.GnBu}
#     cbar_dict = {'TREFHT': None, 'PREC': 3*(10**-7)}
#     key = f"{months}" " " f"{first_year + year}"
      
#     if extrema == True:
#         if dataset == "ds1":
#             t1=9125+98; t2=3650+70
#         elif dataset == "ds0":
#             t1=9125+54; t2=1460+76
#     else:
#         NAO_values, max_dates, min_dates = NAO_max_min_dates(ds, first_year, year_start=year, years=1, months=months)
#         NAO_max_d = max_dates[key]; NAO_min_d = min_dates[key]
#         t1 = NAO_max_d + (year*365); t2 = NAO_min_d + (year*365)

#     # fig, axs = plt.subplots(nrows=2, figsize=(10,10), subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
#     fig, axs = plt.subplots(nrows=2, figsize=(15,8), subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))

#     for ax in axs:
#         ax.coastlines(resolution='110m')
#         gl = ax.gridlines(color='black', linestyle=':', draw_labels=True)
#         gl.top_labels = False
#         gl.right_labels = False
#         if zoom_EU == True:
#             ax.set_extent([-25, 40, 35, 75], crs=ccrs.PlateCarree())
#         else:
#             ax.set_extent([-180, 60, 20, 90], crs=ccrs.PlateCarree())
    
#     if anomalies == True:
#         ds_max_cyclic = make_cyclic(ds[f"{variable}"].isel(time=t1) - ds['PAV'].isel(time=t1), 'lon')
#         ds_min_cyclic = make_cyclic(ds[f"{variable}"].isel(time=t2) - ds['PAV'].isel(time=t2), 'lon')
#         cbar_dict2 = {'TREFHT': [-35, 30], 'PREC': [-4*(10**-7), 9*(10**-7)]}
#         ds_max_cyclic.plot.contourf(ax=axs[0], cmap=cmap_dict[variable], vmin=cbar_dict2[f"{variable}"][0], vmax=cbar_dict2[f"{variable}"][1], levels=14, transform=ccrs.PlateCarree(), cbar_kwargs={"label": "Precipitation anomaly (m/s)"}) 
#         ds_min_cyclic.plot.contourf(ax=axs[1], cmap=cmap_dict[variable], vmin=cbar_dict2[f"{variable}"][0], vmax=cbar_dict2[f"{variable}"][1], levels=14, transform=ccrs.PlateCarree(), cbar_kwargs={"label": "Precipitation anomaly (m/s)"}) 
#     else:
#         ds_max_cyclic = make_cyclic(ds[f"{variable}"].isel(time=t1), 'lon')
#         ds_min_cyclic = make_cyclic(ds[f"{variable}"].isel(time=t2), 'lon')
#         ds_max_cyclic.plot.contourf(ax=axs[0], cmap=cmap_dict[variable], vmax=cbar_dict[f"{variable}"], levels=15, transform=ccrs.PlateCarree()) 
#         ds_min_cyclic.plot.contourf(ax=axs[1], cmap=cmap_dict[variable], vmax=cbar_dict[f"{variable}"], levels=15, transform=ccrs.PlateCarree()) 
    
#     slp_max_cyclic = make_cyclic(ds.SLP.isel(time=t1), 'lon')
#     slp_min_cyclic = make_cyclic(ds.SLP.isel(time=t2), 'lon')
    
#     contours_max = slp_max_cyclic.plot.contour(ax=axs[0], colors='k', levels=15, linewidths=0.7, transform=ccrs.PlateCarree())
#     contours_min = slp_min_cyclic.plot.contour(ax=axs[1], colors='k', levels=15, linewidths=0.7, transform=ccrs.PlateCarree())
#     contours_max.clabel(inline=True, inline_spacing=2, fontsize=8, fmt='%d', colors='black')
#     contours_min.clabel(inline=True, inline_spacing=2, fontsize=8, fmt='%d', colors='black')
                        
#     lons, lats = np.meshgrid(ds0.lon.values, ds0.lat.values)
#     mslp_max = ds.SLP.isel(time=t1).values
#     mslp_min = ds.SLP.isel(time=t2).values
    
#     plot_maxmin_points(axs[0], lons, lats, mslp_max, 'max', 50, symbol='H', color='black')
#     plot_maxmin_points(axs[0], lons, lats, mslp_max, 'min', 25, symbol='L', color='black')
#     plot_maxmin_points(axs[1], lons, lats, mslp_min, 'max', 50, symbol='H', color='black')
#     plot_maxmin_points(axs[1], lons, lats, mslp_min, 'min', 25, symbol='L', color='black')
     
#     axs[0].set_title("highest NAO+ " + f"({str(ds[variable].isel(time=t1).time.values)[:10]})")
#     axs[1].set_title("lowest NAO- " + f"({str(ds[variable].isel(time=t2).time.values)[:10]})")
#     plt.suptitle(f'{period}', fontsize='x-large', fontweight='semibold')
#     plt.tight_layout()
#     return plt.show()

# print(subplots_NAO(dataset='ds0', variable='PREC', year=1, months='Yearly', anomalies=True, extrema=True, zoom_EU=True))

