import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy import interpolate


class Resonances:
    """
    Class for the computation of the various resonances in the machine

    Input:

            OmegaInj (float): frequency of the injected electromagnetic wave
            Eq (object): equilibrium object
            Mesh (object): mesh object
            neFit (function): function specifying the fitting expression to be used for the electron density

    """

    def __init__(self, OmegaInj, Eq, Mesh, neFit):
        self.OmegaInj = OmegaInj
        self.Eq = Eq
        self.Mesh = Mesh
        self.__e = 1.6e-19
        self.__me = 9.11e-31
        self.__mi = 2 * 1.672e-27
        self.neFit = neFit
        self.OmegaCE, self.OmegaUH, self.OmegaLH, self.OmegaPE = Resonances.ComputeResonances(self)
        self.OmegaC1, self.OmegaC2 = Resonances.ComputeCutOff(self)
        self.ZResPosition = Resonances.LocateAxial(self)

    def ComputeResonances(self):
        Br, Bz, e, me, mi, neFit, Eq = (
            self.Eq.Br,
            self.Eq.Bz,
            self.__e,
            self.__me,
            self.__mi,
            self.neFit,
            self.Eq,
        )
        B = np.sqrt(Br**2 + Bz**2)
        OmegaCE = e * B / (2 * np.pi * me)
        OmegaCI = e * B / (2 * np.pi * mi)
        # OmegaPE = 8.98 * np.sqrt(neFit(Eq.R, Eq.Z))
        OmegaPE = 8.98 * np.sqrt(interpolate.splev(Eq.R, neFit))
        plt.contourf(Eq.R[0, :], Eq.Z[:, 0], interpolate.splev(Eq.R, neFit))
        plt.colorbar()
        plt.show()
        OmegaPI = 1 / (2 * np.pi) * np.sqrt(me / mi) * OmegaPE
        OmegaUH = np.sqrt(OmegaPE**2 + OmegaCE**2)
        OmegaLH = np.sqrt(
            OmegaCE * OmegaCI * (1 + OmegaCI**2 / OmegaPI**2) / (1 + OmegaCE**2 / OmegaPE**2)
        )
        return OmegaCE, OmegaUH, OmegaLH, OmegaPE

    def ComputeCutOff(self):
        OmegaCE, OmegaPE = self.OmegaCE, self.OmegaPE
        OmegaC1 = 0.5 * (np.sqrt(4.0 * OmegaPE**2 + OmegaCE**2) - OmegaCE)
        OmegaC2 = 0.5 * (np.sqrt(4.0 * OmegaPE**2 + OmegaCE**2) + OmegaCE)
        return OmegaC1, OmegaC2

    def LocateAxial(self):
        """Method for the computation of the axial location of the resonace"""
        OmegaInj, R, Z, OmegaCE = (
            self.OmegaInj,
            self.Eq.R,
            self.Eq.Z,
            self.OmegaCE,
        )
        cs = plt.contour(R[0, :], Z[:, 0], OmegaCE, [OmegaInj])
        lines = []
        for line in cs.collections[0].get_paths():
            lines.append(line.vertices)
        zpos = np.empty(len(lines))
        for ii in range(0, len(lines)):
            zpos[ii] = np.array(lines[ii])[:, 1].min()
        return zpos

    def PlotResonances(self):
        """Method for plotting of the resonances"""
        OmegaInj, OmegaCE, OmegaUH, OmegaLH, R, Z = (
            self.OmegaInj,
            self.OmegaCE,
            self.OmegaUH,
            self.OmegaLH,
            self.Eq.R,
            self.Eq.Z,
        )
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
        ax[0].contour(R[0, :], Z[:, 0], OmegaCE, [OmegaInj])
        ax[0].set_title(r"$\Omega_{CE}$")
        ax[0].set_xlabel("R [m]")
        ax[0].set_ylabel("Z [m]")
        ax[1].contour(R[0, :], Z[:, 0], OmegaUH, [OmegaInj])
        ax[1].set_title(r"$\Omega_{UH}$")
        ax[1].set_xlabel("R [m]")
        ax[1].set_ylabel("Z [m]")
        ax[2].contour(R[0, :], Z[:, 0], OmegaLH, [OmegaInj])
        ax[2].set_title(r"$\Omega_{LH}$")
        ax[2].set_xlabel("R [m]")
        ax[2].set_ylabel("Z [m]")
        plt.show()
        plt.contourf(R[0, :], Z[:, 0], OmegaUH)
        plt.colorbar()
        fig.tight_layout()

    def PlotCutOff(self):
        """Method for plotting of the cutoffs"""
        OmegaInj, OmegaC1, OmegaC2, R, Z = (
            self.OmegaInj,
            self.OmegaC1,
            self.OmegaC2,
            self.Eq.R,
            self.Eq.Z,
        )
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        ax[0].contour(R[0, :], Z[:, 0], OmegaC1, [OmegaInj])
        ax[0].set_title(r"$\Omega_{C1}$")
        ax[0].set_xlabel("R [m]")
        ax[0].set_ylabel("Z [m]")
        ax[1].contour(R[0, :], Z[:, 0], OmegaC2, [OmegaInj])
        ax[1].set_title(r"$\Omega_{C2}$")
        ax[1].set_xlabel("R [m]")
        ax[1].set_ylabel("Z [m]")
        fig.tight_layout()
        plt.show()
        plt.contourf(R[0, :], Z[:, 0], OmegaC2)
        plt.colorbar()

    def PlotResonancesMesh(self):
        """Method for plotting of the resonances on the physical mesh"""
        OmegaCE, OmegaUH, OmegaLH, R, Z, Grid, n, m, OmegaInj = (
            self.OmegaCE,
            self.OmegaUH,
            self.OmegaLH,
            self.Eq.R,
            self.Eq.Z,
            self.Mesh.Mesh,
            self.Mesh.n,
            self.Mesh.m,
            self.OmegaInj,
        )
        OmegaCEInterp = scipy.interpolate.RectBivariateSpline(Z[:, 0], R[0, :], OmegaCE)
        OmegaUHInterp = scipy.interpolate.RectBivariateSpline(Z[:, 0], R[0, :], OmegaUH)
        OmegaLHInterp = scipy.interpolate.RectBivariateSpline(Z[:, 0], R[0, :], OmegaLH)
        Rc, Zc = Grid[:, 2], Grid[:, 3]
        OmegaCEGrid = np.zeros(n * (m + 2))
        OmegaUHGrid = np.zeros(n * (m + 2))
        OmegaLHGrid = np.zeros(n * (m + 2))
        for ii in range(0, Rc.size):
            OmegaCEGrid[ii] = OmegaCEInterp(Zc[ii], Rc[ii])
            OmegaUHGrid[ii] = OmegaUHInterp(Zc[ii], Rc[ii])
            OmegaLHGrid[ii] = OmegaLHInterp(Zc[ii], Rc[ii])
        OmegaCEGrid = OmegaCEGrid.reshape(m + 2, n)
        OmegaUHGrid = OmegaUHGrid.reshape(m + 2, n)
        OmegaLHGrid = OmegaLHGrid.reshape(m + 2, n)
        it = 0
        Grid = np.delete(Grid, (1), axis=1)
        Grid = np.delete(Grid, (0), axis=1)
        Shape = Grid.shape
        A = np.zeros(Shape[0] * (Shape[1] - 2))
        for ii in range(0, Shape[0]):
            for jj in range(0, Shape[1] - 2):
                A[it] = Grid[ii, jj]
                it = it + 1
        x = A[0 : A.size : 2]
        y = A[1 : A.size : 2]
        Points = n * 5
        Rc = Rc.reshape(m + 2, n)
        Zc = Zc.reshape(m + 2, n)
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
        ax[0].contour(Rc, Zc, OmegaCEGrid, [OmegaInj])
        ax[0].set_title(r"$\Omega_{CE}$")
        ax[0].set_xlabel("R [m]")
        ax[0].set_ylabel("Z [m]")
        for ii in range(0, m + 3):
            ax[0].plot(
                x[ii * n * 5 + 1 : Points], y[ii * n * 5 + 1 : Points], "black", linewidth=0.5
            )
            Points = Points + n * 5
        Points = n * 5
        ax[1].contour(Rc, Zc, OmegaUHGrid, [OmegaInj])
        ax[1].set_title(r"$\Omega_{UH}$")
        ax[1].set_xlabel("R [m]")
        ax[1].set_ylabel("Z [m]")
        for ii in range(0, m + 3):
            ax[1].plot(
                x[ii * n * 5 + 1 : Points], y[ii * n * 5 + 1 : Points], "black", linewidth=0.5
            )
            Points = Points + n * 5
        Points = n * 5
        ax[2].contour(Rc, Zc, OmegaLHGrid, [OmegaInj])
        ax[2].set_title(r"$\Omega_{LH}$")
        ax[2].set_xlabel("R [m]")
        ax[2].set_ylabel("Z [m]")
        for ii in range(0, m + 3):
            ax[2].plot(
                x[ii * n * 5 + 1 : Points], y[ii * n * 5 + 1 : Points], "black", linewidth=0.5
            )
            Points = Points + n * 5
        Points = n * 5
        plt.show()
        fig.tight_layout()
