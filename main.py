""" 
The following simple code represent the basic set of operations one needs to perform for the computation of the magnetic equilibrium as well as the corresponding
field-aligned grid. 
"""
from Equilibrium import *
r1 = 1.0E-8 
r2 = 0.125 
z1 = -1.01 
z2 = 1.01
Nr = 105
Nz = 105
Eq = Equilibrium(r1, r2, z1, z2, Nr, Nz)
PlotEq = PlotEquilibrium(Eq)
PlotEq.PlotPsiContour()
Mesh = Mesh(Eq, 50, 200, 0.125)
Mesh.PlotMesh()
Mesh.WriteMesh()
