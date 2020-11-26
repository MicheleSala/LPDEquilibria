from Equilibrium import Equilibrium, Coils, PlotEquilibrium, Mesh
from scipy import interpolate
r1 = 1.0E-8 
r2 = 0.125 
z1 = -1.01 
z2 = 1.01
Nr = 105
Nz = 105
Eq = Equilibrium(r1, r2, z1, z2, Nr, Nz)
PlotEq = PlotEquilibrium(Eq)
PlotEq.PlotPsiContour()
#PlotEq.PlotBContour()
#PlotEq.PlotPsiBContour()
MESH = Mesh(Eq, 15, 77, 0.125)
#MESH.PlotMesh()

