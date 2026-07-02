from Equilibrium import Equilibrium, Mesh, PlotEquilibrium

r1 = 1.0e-8
r2 = 0.125
z1 = -1.01
z2 = 1.01
Nr = 105
Nz = 105
Eq = Equilibrium(r1, r2, z1, z2, Nr, Nz)
PlotEq = PlotEquilibrium(Eq)
PlotEq.PlotPsiContour()
mesh = Mesh(Eq, 50, 200, 0.125)
mesh.PlotMesh()
mesh.WriteMesh()
