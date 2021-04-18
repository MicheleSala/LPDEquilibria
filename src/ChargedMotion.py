#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:10:38 2020

@author: michele
"""
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from Equilibrium import *
from scipy import interpolate
matplotlib.rcParams.update({'font.size': 24, 'font.family': 'serif', 'mathtext.fontset': 'stix'})

class Particle:
    """
    Class for the representation of the charged particle to be tracked
    """
    def __init__(self, Z, m):
        """
        Constructor function
        
        Input:
            Z (int) atomic number of the particle
            m (float) mass of the particle
        """
        self.__e = 9.11E-31
        self.q = Z * self.__e
        self.m = m
        self.Omega = self.q / self.m 
    def updateParticleState(self, x, v):
        """
        Method for updating the particle position
        Input:
            x0 (float) initial position of the particle
        Return: 
            x (float) final position of the particle
            
        """
        self.x0 = x
        self.v0 = v
    
    def initializeParticleState(self, x0, v0):
        """
        Method for the initialization of the particle state
        """
        self.x0 = x0
        self.v0 = v0
    
    def initializeSimulationDomain(self, xmin, xmax, ymin, ymax):
        """
        Function for the initialization of the simulation domain 
        """
        self.xmin = xmin 
        self.xmax = xmax
        self.ymin = ymin 
        self.ymax = ymax
        
    def checkBoundaries(self):
        """
        Method for applying the boundary conditions 
        """
        if x0[0, 0] >= xmax:
        self.initializeParticleState(x0[0, 0], -v0[0, 0]
        if x0[0, 0] <= xmin:
        x0[0, 0] = xmin
        v0[0, 0] = -v0[0, 0]
        if x0[1, 0] >= ymax:
        x0[1, 0] = ymax
        v0[1, 0] = -v0[1, 0]
        if x0[1, 0] <= ymin:
        x0[1, 0] = ymin
        v0[1, 0] = -v0[1, 0] 
    
    def printProperties(self):
        """
        Function for prnting the particle properties, i.e. charge and mass
        """
        print('Particle charge {}: \nParticle mass {}'.format(self.q, self.m))

class MagneticField(Equilibrium):
    """
    Class for the representation of the magnetic field. Child class of the Equilibrum one
    """
    def __init__(self, r1, r2, z1, z2, Nr, Nz):
        """
        Initialize the equilibrium 
        """
        Equilibrium.__init__(self, r1, r2, z1, z2, Nr, Nz)
    
    def InterpField(self, x0):
        """
        Function for the interpolation of the magnetic field at the position x0 
        """
        Bx, By, Z, R = self.Br, self.Bz, self.Z, self.R
        BxInterp = scipy.interpolate.RectBivariateSpline(Z[:, 0], R[0, :], Bx)
        ByInterp = scipy.interpolate.RectBivariateSpline(Z[:, 0], R[0, :], By)
        BxPos = BxInterp(x0[1, 0], x0[0, 0])
        ByPos = ByInterp(x0[1, 0], x0[0, 0])
        return BxPos, ByPos
    
        
        
        

    
# Charged particle motion in a given magnetic field

r1 = 1.0E-8 
r2 = 0.125 
z1 = 0.5 
z2 = 2.5
Nr = 105
Nz = 105
Eq = Equilibrium(r1, r2, z1, z2, Nr, Nz)
PlotEq = PlotEquilibrium(Eq)
PlotEq.PlotPsiContour()
X = Eq.R
Y = Eq.Z

def Step(x0, v0, dt):
    """
    Function for the advancment of the particle position
    """
    xmax = 0.125
    xmin = 0.0
    ymax = 2.5
    ymin = 0.5
    e = -1.6E-19
    me = 9.11E-25
    Omega = e * dt / me
   # if (x0[0, 0] >= xmax or x0[0, 0] <= xmin or x0[1, 0] >= ymax or x0[1, 0] <= ymin):
    if x0[0, 0] >= xmax:
       x0[0, 0] = xmax
       v0[0, 0] = -v0[0, 0]
    if x0[0, 0] <= xmin:
        x0[0, 0] = xmin
        v0[0, 0] = -v0[0, 0]
    if x0[1, 0] >= ymax:
        x0[1, 0] = ymax
        v0[1, 0] = -v0[1, 0]
    if x0[1, 0] <= ymin:
        x0[1, 0] = ymin
        v0[1, 0] = -v0[1, 0] 
    
    #if (x0[0, 0] >= xmax or x0[0, 0] <= xmin or x0[1, 0] >= ymax or x0[1, 0] <= ymin):
     #   return;
    Bx, By = InterpField(Eq, x0)
    Bz = np.array(0.)
    E = np.array([[0., 0., 0.]]).T
    P = np.array(((0., Bz, -By[0, 0]), (-Bz, 0., Bx[0, 0]), (By[0, 0], -Bx[0, 0], 0.)))
#    v = np.linalg.inv(np.eye(3) + Omega * P / 2.).dot(np.eye(3) - Omega * P / 2.).dot(v0)
    v = np.linalg.inv(np.eye(3) + Omega * P / 2.).dot(np.eye(3) - Omega * P / 2.).dot(v0) + np.linalg.inv(np.eye(3) + Omega * P / 2.).dot(e / me * dt * E)
    x = x0 + v * dt
    return x, v

v0 = np.array([[10., 10., 0.]]).T
x0 = np.array([[0.025, 1.5, 0.]]).T
dt = 1.0E-04
Nstep = 6000
Npart = 1
fig = plt.figure()
ax = fig.gca(projection='3d')
ax = fig.gca()
xx = np.empty((Nstep, Npart))
xy = np.empty((Nstep, Npart))
xz = np.empty((Nstep, Npart))
vx = np.empty((Nstep, Npart))
vy = np.empty((Nstep, Npart))
vz = np.empty((Nstep, Npart))
for jj in range(0, Npart):
    v0 = np.array([[1., 1., 10.]]).T
    x0 = np.array([[0.001, 1.25, 0.]]).T
    for ii in range(0, Nstep):
        x, v = Step(x0, v0, dt)
        x0 = x
        v0 = v
        xx[ii, jj] = x[0]
        xy[ii, jj] = x[1]
        xz[ii, jj] = x[2]
        vx[ii, jj] = v[0]
        vy[ii, jj] = v[1]
        vz[ii, jj] = v[2]
for jj in range(0, Npart):    
    ax.plot(xx[:, jj], xy[:, jj], xz[:, jj], linewidth = 0.5)
#ax.contour(Eq.R, Eq.Z, Eq.psi)
plt.show()
r = np.sqrt(xx ** 2 + xz ** 2)
theta = np.arctan2(xz, xx)
fig = plt.figure()
plt.polar(xz * 180. / np.pi, xx, linewidth = 0.1)
plt.show()
"""
Plot bello
"""
fig = plt.figure()
ax = fig.gca()
for jj in range(0, Npart):
    ax.plot(xy[:, jj], xx[:, jj], 'grey', linewidth = 0.3) # Plot particle position in the x,y plane
#ContourLevels = np.linspace(np.sqrt(np.abs(Eq.psi.min())), np.sqrt(Eq.psi.max()), 50)
ContourLevels = np.linspace(np.sqrt(np.abs(Eq.psi.min())), 0, 25)
ContourLevelsPos = np.linspace(0, np.sqrt(Eq.psi.max()), 25)
cnt = ax.contour(Eq.R, Eq.Z, Eq.psi, -ContourLevels ** 2, cmap=matplotlib.cm.RdGy, linewidths = 1.0)
cnt = ax.contour(Eq.R, Eq.Z, Eq.psi, ContourLevelsPos ** 2, cmap=matplotlib.cm.RdGy, linewidths = 1.0)
plt.show()

# 3d plot trajectory
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.grid(False)
for jj in range(0, Npart):    
    ax.plot(xx[:, jj], xy[:, jj], xz[:, jj], 'grey', linewidth = 0.5)
ContourLevels = np.linspace(np.sqrt(np.abs(Eq.psi.min())), 0, 25)
ContourLevelsPos = np.linspace(0, np.sqrt(Eq.psi.max()), 25)
cnt = ax.contour(Eq.R, Eq.Z, Eq.psi, -ContourLevels ** 2, cmap=matplotlib.cm.RdGy, linewidths = 0.8)
cnt = ax.contour(Eq.R, Eq.Z, Eq.psi, ContourLevelsPos ** 2, cmap=matplotlib.cm.RdGy, linewidths = 0.8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()
# Guiding centre
B = np.array([Eq.Br, Eq.Bz, 0.])
b = B / np.linalg.norm(B)
e = -1.6E-19
me = 9.11E-25
Omega = e * B / me
R = x - (1. / Omega) * np.cross(b, v)



