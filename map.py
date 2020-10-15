"""
Produces a map showing London Underground station locations with high
resolution background imagery provided by OpenStreetMap.

"""
import argparse
import fnmatch
import matplotlib.pyplot as plt
import os
import pandas as pd

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM

meas_cols = ['NO2 (Plume AQI)', 'NO2 (ppb)', 'VOC (Plume AQI)', 'VOC (ppb)',
             'pm 10 (Plume AQI)', 'pm 10 (ug/m3)', 'pm 2.5 (Plume AQI)',
             'pm 2.5 (ug/m3)']
aqi_cols = ['NO2 (Plume AQI)', 'VOC (Plume AQI)', 'pm 10 (Plume AQI)',
            'pm 2.5 (Plume AQI)']

def load_data(folder):
    # Load position data
    matches = [fn for fn in os.listdir(folder) if
               fnmatch.fnmatch(fn,'user_positions*.csv')]
    pos_fn = os.path.join(folder,matches[0])
    pos_df = pd.read_csv(pos_fn)

    # Load measurement data
    matches = [fn for fn in os.listdir(folder) if
               fnmatch.fnmatch(fn,'user_measures*.csv')]
    meas_fn = os.path.join(folder,matches[0])
    meas_df = pd.read_csv(meas_fn)

    return pos_df,meas_df

def merge_data(pos_df,meas_df):
    # Merge data (thus discarding any without locations)
    # For each pos datum, find the measurements within t_tol seconds for which that
    # is the nearest pos datum, temporally:
    t_tol = 10
    win0 = 0
    win1 = 0
    df = pd.DataFrame()
    for index,pos_row in pos_df.iterrows():
        # Move window start to next row within t_tol seconds:
        while win0<len(meas_df.index) and \
                pos_row.timestamp-meas_df.timestamp.iloc[win0]>t_tol:
            win0 += 1
        # Move window end to last row within t_tol seconds:
        legit = False
        if win0<len(meas_df.index):
            while win1<len(meas_df.index) and \
                    meas_df.timestamp.iloc[win1]-pos_row.timestamp<t_tol:
                legit = True
                win1 += 1
        # Take average measurement in window
        if legit:
            avg_meas = meas_df.iloc[win0:win1].mean()
            avg_meas = avg_meas.drop(['timestamp'])
            new_row = pd.concat([pos_row,avg_meas],axis=0)
            df = df.append(new_row,ignore_index=True)
        # Safeguard against overshoot
        win1 -= 1

    return df

def process_data(df,args):

    # Add overall AQI
    df['AQI'] = df[aqi_cols].max(axis=1)

    if args.medadjust:
        # Adjust data for local median
        t_tol = 300
        win0 = 0
        win1 = 0
        for index, row in df.iterrows():
            # Move window start to next row within t_tol seconds:
            while row.timestamp - df.timestamp.iloc[win0] > t_tol:
                win0 += 1
            if win0>len(df.index):
                break
            # Move window end to last row within t_tol seconds:
            legit = False
            if win0<len(df.index):
                while win1<len(df.index) and \
                        df.timestamp.iloc[win1] - row.timestamp < t_tol:
                    legit = True
                    win1 += 1
            # Subtract median
            if legit:
                median = df[meas_cols].iloc[win0:win1].median()
                df.loc[index,meas_cols] = row[meas_cols].subtract(median)
            else:
                df = df.drop(index=index)
            # Safeguard against overshoot
            win1 -= 1

    return df


def plot_data(df,args):
    imagery = OSM()

    ax = plt.axes(projection=imagery.crs)
    ax.set_extent((df.longitude.min(), df.longitude.max(),
                   df.latitude.min(), df.latitude.max()))

    # Add the imagery to the map.
    ax.add_image(imagery, 14)

    # Plot data
    plotcol = args.plotcol
    for col in df.columns:
        if args.plotcol in col:
            plotcol = col
    sc = plt.scatter(df.longitude,df.latitude,c=df[plotcol],
                     transform=ccrs.PlateCarree())
    plt.title(plotcol)
    plt.clim(-10,50)
    plt.colorbar(sc)
    plt.show()


parser = argparse.ArgumentParser()

parser.add_argument("--folder", default='data',
                    help="Relative path to a directory containing your exported flow data.")
parser.add_argument("--plotcol", default='AQI',
                    help="Which data column to plot")
parser.add_argument("--medadjust", dest='medadjust', action='store_true',
                    help="Adjust data by subtracting median")

if __name__ == '__main__':

    args = parser.parse_args()

    # Load data
    import time
    print(f"Loading data from {args.folder}...")
    pos_data, meas_data = load_data(args.folder)
    print(f"Merging data...")
    data = merge_data(pos_data, meas_data)

    # Process data
    print(f"Processing data...")
    data = process_data(data,args)

    # Plot on map
    print("Making map...")
    plot_data(data,args)