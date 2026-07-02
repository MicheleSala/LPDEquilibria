import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import scipy.interpolate


class Mesh:
    """
    Class for the definition and construction of a field-aligned structured Mesh

    Input:
            Eq (objtect): equilibrium object defining the magnetic equilibrium
            m (int): number of point in the radial direction
            n (int): number of point in the axial direction
            Rlim (float): location in the r direction of the limiting surface

    Return:
            Mesh (object)
    """

    def __init__(self, Eq, m, n, Rlim):
        self.Eq = Eq
        self.m = m
        self.n = n
        self.__Ncont = 100
        self.__LowerPsiValue = 1.0e-08
        self.__Dim = 24
        self.Rlim = Rlim
        PsiLim = Mesh.ComputeLastPsi(self)
        Mesh.PsiLim = PsiLim
        self.Mesh = Mesh.ComputeGrid(self)

    def ComputeLastPsi(self):
        """
        Function for the computation of the last magnetic flux surface
        """
        psi, R, Z, LowerPsiValue, Ncont, Rlim = (
            self.Eq.psi,
            self.Eq.R,
            self.Eq.Z,
            self.__LowerPsiValue,
            self.__Ncont,
            self.Rlim,
        )
        ContourLevels = np.linspace(LowerPsiValue, psi.max(), Ncont)
        Cnt = plt.contour(R, Z, psi, ContourLevels)
        lines = []
        for it in range(0, Ncont):
            for line in Cnt.collections[it].get_paths():
                lines.append(line.vertices)
            r = np.array(lines[it])[:, 0].max() - Rlim
            if r >= 0:
                print("Found LCFS at: ", ContourLevels[it - 1])
                break
        return ContourLevels[it - 1]

    def InterpolatePsi(self, zP, rP, zQuery, ContourLevels):
        """
        Function for the interpolation of the psi function
        """
        psi = self.Eq.psi
        for ii in range(0, ContourLevels.size):
            Cnt = np.array(plt.contour(rP, zP, psi, [ContourLevels[ii]]).allsegs)
            plt.ioff()
            Shape = Cnt.shape
            r = Cnt.reshape(Shape[2], 2)[:, 0]
            z = Cnt.reshape(Shape[2], 2)[:, 1]
            InterpFunc = scipy.interpolate.interp1d(z, r, fill_value="extrapolate")
            if ii == 0:
                InterpPsi = InterpFunc(zQuery).reshape(1, zQuery.size)
            else:
                InterpPsi = np.append(InterpPsi, InterpFunc(zQuery).reshape(1, zQuery.size), axis=0)
        return InterpPsi.T

    def InterpolateField(self, rQuery, MESH):
        Bz, R, Z, n, m = self.Eq.Bz, self.Eq.R, self.Eq.Z, self.n, self.m
        zQuery = MESH[:, 3]
        BzC = np.zeros(n * (m + 2))
        InterpFunc = scipy.interpolate.RectBivariateSpline(Z[:, 0], R[0, :], Bz)
        for ii in range(0, zQuery.size):
            BzC[ii] = InterpFunc(zQuery[ii], rQuery[ii])
        return BzC

    def ComputeGrid(self):
        LowerPsiValue, PsiLim, m, n, Z, R, z2 = (
            self.__LowerPsiValue,
            self.PsiLim,
            self.m,
            self.n,
            self.Eq.Z,
            self.Eq.R,
            self.Eq.z2,
        )
        ContourLevels = np.linspace(np.sqrt(LowerPsiValue), np.sqrt(PsiLim), m) ** 2
        zValues = Z[:, 0]
        rValues = R[0, :]
        alpha = 1.8
        overshoot = 1.0e-05

        def func(x):
            return z2 * ((np.tanh(-x)) / (np.tanh(alpha)) * (1 + overshoot)) / (1 + overshoot)

        IndexFunc = np.linspace(-alpha, alpha, 2 * n + 1)
        zQuery = func(IndexFunc)
        InterpPsi = Mesh.InterpolatePsi(self, zValues, rValues, zQuery, ContourLevels)
        crxT = zQuery[0 : zQuery.size - 1 : 2]
        crxB = zQuery[2 : zQuery.size : 2]
        cryLT = InterpPsi[0 : InterpPsi.shape[0] - 1 : 2, 0 : InterpPsi.shape[1] - 1].T.reshape(
            1, n * (m - 1)
        )
        cryRT = InterpPsi[0 : InterpPsi.shape[0] - 1 : 2, 1 : InterpPsi.shape[1]].T.reshape(
            1, n * (m - 1)
        )
        cryLB = InterpPsi[2 : InterpPsi.shape[0] : 2, 0 : InterpPsi.shape[1] - 1].T.reshape(
            1, n * (m - 1)
        )
        cryRB = InterpPsi[2 : InterpPsi.shape[0] : 2, 1 : InterpPsi.shape[1]].T.reshape(
            1, n * (m - 1)
        )
        cryBottomGuard = np.zeros((1, n))
        cryUpperGuard = Mesh.InterpolatePsi(self, zValues, rValues, zQuery, np.array([1.0e-12]))
        cryUpperGuardLT = cryUpperGuard[0 : cryUpperGuard.size - 1 : 2]
        cryUpperGuardLB = cryUpperGuard[2 : cryUpperGuard.size : 2]
        cryLT = np.append(np.append(cryBottomGuard, cryUpperGuardLT), cryLT)
        cryLB = np.append(np.append(cryBottomGuard, cryUpperGuardLB), cryLB)
        cryRT = np.append(np.append(cryUpperGuardLT, cryLT[2 * n : 3 * n]), cryRT)
        cryRB = np.append(np.append(cryUpperGuardLB, cryLB[2 * n : 3 * n]), cryRB)
        cryLB = np.append(cryLB, cryRB[cryRB.size - n : cryRB.size])
        cryLT = np.append(cryLT, cryRT[cryRB.size - n : cryRB.size])
        cryRT = np.append(cryRT, cryRT[cryRB.size - n : cryRB.size] + 1.0e-05)
        cryRB = np.append(cryRB, cryRB[cryRB.size - n : cryRB.size] + 1.0e-05)
        cryC = (cryLT + cryRB) / 2
        crxC = (crxT + crxB) / 2
        MESH = np.empty((n * (m + 2), 14))
        MESH[:, 2] = cryC
        MESH[:, 3] = np.matlib.repmat(crxC, 1, m + 2)
        MESH[:, 4] = cryLT
        MESH[:, 5] = np.matlib.repmat(crxT, 1, m + 2)
        MESH[:, 6] = cryLB
        MESH[:, 7] = np.matlib.repmat(crxB, 1, m + 2)
        MESH[:, 8] = cryRT
        MESH[:, 9] = np.matlib.repmat(crxT, 1, m + 2)
        MESH[:, 10] = cryRB
        MESH[:, 11] = np.matlib.repmat(crxB, 1, m + 2)
        MESH[:, 0] = np.matlib.repmat(np.array([range(0, n)]), 1, m + 2)
        MESH[:, 1] = np.matlib.repeat(np.array([range(0, m + 2)]), n)
        BzC = Mesh.InterpolateField(self, cryC, MESH)
        MESH[:, 12] = BzC.reshape(n * (m + 2))
        MESH[:, 13] = 0
        return MESH

    def PlotMesh(self):
        Mesh = self.Mesh
        it = 0
        Mesh = np.delete(Mesh, (1), axis=1)
        Mesh = np.delete(Mesh, (0), axis=1)
        Shape = Mesh.shape
        A = np.zeros(Shape[0] * (Shape[1] - 2))
        for ii in range(0, Shape[0]):
            for jj in range(0, Shape[1] - 2):
                A[it] = Mesh[ii, jj]
                it = it + 1
        x = A[0 : A.size : 2]
        y = A[1 : A.size : 2]
        Points = self.n * 5
        fig, ax = plt.subplots()
        for ii in range(0, self.m + 3):
            ax.plot(
                x[ii * self.n * 5 + 1 : Points],
                y[ii * self.n * 5 + 1 : Points],
                "black",
                linewidth=0.5,
            )
            Points = Points + self.n * 5
        ax.set_aspect(aspect=0.3)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        plt.show()

    def WriteMesh(self):
        Mesh = self.Mesh
        Name = input("Insert Mesh name: ") + ".ASCII"
        pathName = "meshes/"
        np.savetxt(pathName + Name, Mesh, "%.8E")
        print("Saved Mesh in file: " + Name)
