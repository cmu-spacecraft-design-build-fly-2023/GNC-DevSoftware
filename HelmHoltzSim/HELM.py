import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ppigrf
from datetime import datetime

def helmplot():
    Ix = 0.244 #current in x axis
    Iy = 0.039 #current in y axis
    Iz = 0.56 #current in z axis
    Nx = 50 #number of loops
    Ny = 50
    Nz = 50
    Rx = 0.40 #radius
    Ry = 0.38
    Rz = 0.36
    dx = 0.38 #distance between the coils
    dy = 0.36
    dz = 0.34
    mu0 = 4*np.pi*10**-7
    
    B1x = [None]*2000
    B2x = [None]*2000
    B1y = [None]*2000
    B2y = [None]*2000
    B1z = [None]*2000
    B2z = [None]*2000
    Btotx = [None]*2000
    Btoty = [None]*2000
    Btotz = [None]*2000
    z = list(np.arange(-1.0,1.0,0.001))
    

    
    for i in range(0,2000):
        #B1x[i] = ((0.5*mu0)*np.power((Nx*Ix*Rx*Rx)/(Rx*Rx+np.power((z[i]+dx/2),2)),(1.5)))*(10**9)
        B1x[i] = ((0.5*mu0)*((Nx*Ix*Rx*Rx)/(Rx*Rx+(z[i]+dx/2)**2)**(3/2)))*(10**9)
        B2x[i] = ((0.5*mu0)*((Nx*Ix*Rx*Rx)/(Rx*Rx+(z[i]-dx/2)**2)**(3/2)))*(10**9)
        
        B1y[i] = ((0.5*mu0)*((Ny*Iy*Ry*Ry)/(Ry*Ry+(z[i]+dy/2)**2)**(3/2)))*(10**9)
        B2y[i] = ((0.5*mu0)*((Ny*Iy*Ry*Ry)/(Ry*Ry+(z[i]-dy/2)**2)**(3/2)))*(10**9)

        B1z[i] = ((0.5*mu0)*((Nz*Iz*Rz*Rz)/(Rz*Rz+(z[i]+dz/2)**2)**(3/2)))*(10**9)
        B2z[i] = ((0.5*mu0)*((Nz*Iz*Rz*Rz)/(Rz*Rz+(z[i]-dz/2)**2)**(3/2)))*(10**9)

        #LEO is between 25000-65000nT
        Btotx[i] = B1x[i]+B2x[i]
        Btoty[i] = B1y[i]+B2y[i]
        Btotz[i] = B1z[i]+B2z[i]
        
        
    plt.subplot(3,1,1)
    plt.plot(z,B1x, label = "Left loop")
    plt.plot(z,B2x, label = "Right loop")
    plt.plot(z,Btotx, label = "Coil")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.title("Magnetic field strength along three axis")
    plt.xlabel("x-axis (m)")
    plt.ylabel("Magnetic field strength (nT)")
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(z,B1y, label = "Left loop")
    plt.plot(z,B2y, label = "Right loop")
    plt.plot(z,Btoty, label = "Coil")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel("y-axis (m)")
    plt.ylabel("Magnetic field strength (nT)")
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(z,B1z, label = "Left loop")
    plt.plot(z,B2z, label = "Right loop")
    plt.plot(z,Btotz, label = "Coil")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel("z-axis (m)")
    plt.ylabel("Magnetic field strength (nT)")
    plt.legend()
    
    
    
    plt.show()
    
def currentCalc():
    lon = -79.995888  # degrees east
    lat = 40.440624 # degrees north
    h = 0.4178808        # kilometers above sea level
    date = datetime(2023, 3, 28)
    leox = 25000
    leoy = 25000
    leoz = 25000
    
    Nx = 50 #number of loops
    Ny = 50
    Nz = 50
    Rx = 0.40 #radius
    Ry = 0.38
    Rz = 0.36
    dx = 0.38 #distance between the coils
    dy = 0.36
    dz = 0.34
    mu0 = 4*np.pi*10**-7

    Be, Bn, Bu = ppigrf.igrf(lon, lat, h, date) # returns east(+E|-W), north(+N|-S), up(-D|+U) in nT
    #print(Be)
    #print(Bn)
    #print(Bu)
    
    Bx = -Be[0] + leox
    By = -Bn[0] + leoy
    Bz = -Bu[0] + leoz
    
    #print(tar_Magx)
    #print(tar_Magy)
    #print(tar_Magz)
    

    coilx = 2/(((Rx**2)+((dx/2)**2))**1.5)
    ix = ((2*Bx)/(mu0*coilx*Nx*(Rx**2)))/(10**9)
    print(ix)
    
    coily = 2/(((Ry**2)+((dy/2)**2))**1.5)
    iy = ((2*By)/(mu0*coily*Ny*(Ry**2)))/(10**9)
    print(iy)
    
    coilz = 2/(((Rz**2)+((dz/2)**2))**1.5)
    iz = ((2*Bz)/(mu0*coilz*Nz*(Rz**2)))/(10**9)
    print(iz)

if __name__ == "__main__":
    currentCalc()
    helmplot()
    