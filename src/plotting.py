import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 18, "font.family": "serif", "mathtext.fontset": "stix"})


class PlotEquilibrium:
    """
    Class for plotting the Equilibrium object
    """

    def __init__(self, Eq, Ncont=100, Dim=24):
        self.Equ = Eq
        self.Ncont = Ncont
        self.Dim = Dim

    def PlotPsiContour(self):
        """
        Contour plot of psi
        """
        R, Z, psi, Ncont = self.Equ.R, self.Equ.Z, self.Equ.psi, self.Ncont
        fig, ax = plt.subplots()
        cnt = ax.contour(
            R,
            Z,
            psi,
            Ncont,
            cmap=matplotlib.cm.RdGy,
            vmin=abs(self.Equ.psi).min(),
            vmax=abs(self.Equ.psi).max(),
        )
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        fig.colorbar(cnt, ax=ax)
        # plt.show()

    def PlotBContour(self):
        """
        Contour plot of B
        """
        R, Z, Ncont = self.Equ.R, self.Equ.Z, self.Ncont
        B = np.sqrt(self.Equ.Br**2 + self.Equ.Bz**2)
        fig, ax = plt.subplots()
        cnt = ax.contourf(
            R, Z, B, Ncont, cmap=matplotlib.cm.RdGy, vmin=abs(B).min(), vmax=abs(B).max()
        )
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        fig.colorbar(cnt, ax=ax)
        plt.show()

    def PlotPsiBContour(self):
        """
        Plot psi and B contour on the same figure
        """
        R, Z, psi, Ncont = (
            self.Equ.R,
            self.Equ.Z,
            self.Equ.psi,
            self.Ncont,
        )
        B = np.sqrt(self.Equ.Br**2 + self.Equ.Bz**2)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        cntPsi = ax[0].contour(
            R[0, :],
            Z[:, 0],
            psi,
            Ncont,
            cmap=matplotlib.cm.RdGy,
            vmin=abs(psi).min(),
            vmax=abs(psi).max(),
        )
        fig.colorbar(cntPsi, ax=ax[0])
        ax[0].set_xlabel("R [m]")
        ax[0].set_ylabel("Z [m]")
        cntB = ax[1].contourf(
            R, Z, B, Ncont, cmap=matplotlib.cm.RdGy, vmin=abs(B).min(), vmax=abs(B).max()
        )
        fig.colorbar(cntB, ax=ax[1])
        ax[1].set_xlabel("R [m]")
        ax[1].set_ylabel("Z [m]")
        plt.show()
