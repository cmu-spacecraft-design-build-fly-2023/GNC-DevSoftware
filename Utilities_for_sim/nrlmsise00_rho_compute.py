from spaceweather import sw_daily
import pandas as pd
from datetime import datetime
import numpy as np
from nrlmsise00 import msise_flat


def retrieve_sw_data(epoch_dt):
    # Load space weather data
    sw = sw_daily()

    # Check and set the timezone to UTC if it's not already
    if sw.index.tz is None:
        sw = sw.tz_localize("utc")

    epoch = pd.to_datetime(epoch_dt, utc=True)

    # Calculate the previous day for f10.7
    epoch_prev = epoch - pd.to_timedelta("1d")

    # Retrieve the specific data
    ap = sw.at[epoch.floor("D"), "Apavg"]
    f107 = sw.at[epoch_prev.floor("D"), "f107_obs"]
    f107a = sw.at[epoch.floor("D"), "f107_81ctr_obs"]

    return ap, f107, f107a

def compute_rho(epoch_dt, alt, lat, lon,f107a, f107, ap):
    rho = msise_flat(epoch_dt, 400, 60, -70, f107a, f107, ap, method='gt7d')[5]
    rho = rho*1000
    return rho

# Temporary local testing
if __name__ == "__main__":
    epoch_dt = datetime(2000, 11, 26, 12, 0, 5, 0)
    ap, f107, f107a = retrieve_sw_data(epoch_dt)
    alt = 550
    lat = 60
    lon = -70
    rho = compute_rho(epoch_dt, alt, lat, lon,f107a, f107, ap)
    print('rho', rho)