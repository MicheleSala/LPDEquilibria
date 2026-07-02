import numpy as np


class Sources:
    """Class for the definition of the source input"""

    def __init__(self, RadialDensity, AxialDensity, Mesh, Resonances, sigma):
        self.Grid = Mesh
        self.Res = Resonances
        self.zmin = Resonances.ZResPosition - sigma
        self.zmax = Resonances.ZResPosition + sigma
        self.RadialDensity = RadialDensity
        self.AxialDensity = AxialDensity
        self.RadialPoints = Sources.ComputeRadialPoints(self)
        self.AxialPoints = Sources.ComputeAxialPoints(self)

    def ComputeRadialPoints(self):
        Mesh, n, zmin, zmax = (
            self.Grid.Mesh,
            self.Grid.n,
            self.zmin,
            self.zmax,
        )
        Shape = np.shape(Mesh)
        RcentrePoints = Mesh[0 : Shape[0], 2]
        ZcentrePoints = Mesh[0 : Shape[0], 3]
        ZonAxis = ZcentrePoints[0:n]
        IndInf = np.max(np.where(ZonAxis >= zmin))
        IndMax = np.min(np.where(ZonAxis <= zmax))
        print(IndInf)
        zMinSor = ZonAxis[IndInf]
        zMaxSor = ZonAxis[IndMax]
        IndInf = np.where(zMinSor == ZcentrePoints)
        IndSup = np.where(zMaxSor == ZcentrePoints)
        RadialPoints = (RcentrePoints[IndInf] + RcentrePoints[IndSup]) / 2
        RadialPoints = np.append(-RadialPoints[0], RadialPoints)
        return RadialPoints

    def ComputeAxialPoints(self):
        n = self.Grid.n
        AxialPoints = np.linspace(0, 1, n)
        return AxialPoints

    def WriteSource(self):
        # TODO: complete WriteSource implementation (writes source term to file for edge plasma codes)
        RadialPoints, AxialPoints = self.RadialPoints, self.AxialPoints
        RadialValues = self.RadialDensity(RadialPoints)  # noqa: F841
        AxialValues = self.AxialDensity(AxialPoints)  # noqa: F841
        l = np.size(RadialPoints)  # noqa: F841
