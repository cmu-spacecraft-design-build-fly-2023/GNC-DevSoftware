import numpy as np
import math
import matplotlib.pyplot as plt



""""
Steps to calculate the time a satellite will be in view above a ground station given the 
satellite's altitude:

1. Calculate Earth Angular radius --> The angular radius of the Earth as seen from the satellite
2. Calculate satellite time period (time to orbit Earth once)
3. Calculate maximum nadir angle --> Measured at the satellite from nadir (point directly below) to the ground station
4. Calculate max. Earth centered angle
5. Cal min. Earth centered angle
6. Calculate Time in View
"""


def plot_chart(elevation_angles, view_times):

    plt.plot(elevation_angles, view_times, linestyle='-', marker='o', color='blue', label='View Times')

    plt.xlabel('Satellite Elevation Angle (degrees)', fontsize=10)
    plt.ylabel('Satellite View Time (min)', fontsize=10)
    plt.title('Satellite View Time vs Elevation Angle for Ground Station in Pittsburgh', fontsize=10, fontweight='bold')
    
    plt.show()


def earth_angular_radius(earth_radius, sat_altitude):

    p = math.degrees(math.asin(earth_radius / (earth_radius + sat_altitude)))

    return p


def time_period(earth_mu, earth_radius, sat_altitude):

    semi_major_axis = earth_radius + sat_altitude

    time_period = 2 * np.pi / (np.sqrt(earth_mu)) * semi_major_axis ** (3 / 2)

    return time_period / 60


def max_nadir_angle(p, elevation_min):

    p_rad = math.radians(p)
    elevation_min_rad = math.radians(elevation_min)

    n_max = math.degrees(math.asin(math.sin(p_rad) * math.cos(elevation_min_rad)))

    return n_max



def max_earth_centered_angle(elevation_min, n_max):

    lambda_max = 90 - elevation_min - n_max

    return lambda_max



def min_earth_centered_angle(lat_pole, long_pole, lat_gs, long_gs):

    delta_long = long_gs - long_pole
    delta_long_rad = math.radians(delta_long)
    lat_pole_rad = math.radians(lat_pole)
    lat_gs_rad = math.radians(lat_gs)

    lambda_min = math.degrees(math.asin(math.sin(lat_pole_rad) * math.sin(lat_gs_rad)) + 
                                math.cos(lat_pole_rad) * math.cos(lat_gs_rad) * math.cos(delta_long_rad))

    return lambda_min


def time_in_view(P, lambda_max, lambda_min):

    t_view = (P/180) * math.degrees(math.acos(math.cos(math.radians(lambda_max))/ math.cos(math.radians(lambda_min))))

    return t_view



def view_time_algorithm(earth_radius, sat_altitude, elevation_min, earth_mu, lat_pole, long_pole, lat_gs, long_gs):

    # Earth Angular radius
    p = earth_angular_radius(earth_radius, sat_altitude)

    # satellite time period
    period = time_period(earth_mu, earth_radius, sat_altitude)

    # maximum nadir angle
    n_max = max_nadir_angle(p, elevation_min)

    # max. Earth centered angle
    lambda_max = max_earth_centered_angle(elevation_min, n_max)

    # min. Earth centered angle
    lambda_min = min_earth_centered_angle(lat_pole, long_pole, lat_gs, long_gs)

    # View Time for Groundstation
    view_time = time_in_view(period, lambda_max, lambda_min)

    return view_time



if __name__ == "__main__":

    earth_radius = 6378.0
    sat_altitude = 600
    earth_mu     = 3.9860043543609598E+05 # km^3 / s^2
    elevation_min = 5       # degrees

    lat_pole = 8         # Assuming SSO inclination of 98 degrees
    long_pole = 170
    lat_gs = 40.4432        # Assuming ground staton at CMU, Pittsburgh
    long_gs = 79.9428

    # # Example from SMAD - Page 184, Table 8-11
    # lat_pole = 40         # Assuming SSO inclination of 98 degrees
    # long_pole = 150
    # lat_gs = 33.5        # Assuming ground staton at CMU, Pittsburgh
    # long_gs = 248



    view_time = view_time_algorithm(earth_radius, sat_altitude, elevation_min, earth_mu, lat_pole, long_pole, lat_gs, long_gs)

    print("VIEW TIME", view_time)


    view_times = np.zeros(8)
    elevation_angles = np.zeros(8)

    count = 0
    for i in range(np.size(view_times)):
            
        print(count) 

        view_time = view_time_algorithm(earth_radius, sat_altitude, elevation_min, earth_mu, lat_pole, long_pole, lat_gs, long_gs)

        view_times[count] = view_time
        elevation_angles[count] = elevation_min

        elevation_min += 5

        count += 1
        

    print("View Times\n")
    print(view_times)

    plot_chart(elevation_angles, view_times)
    
