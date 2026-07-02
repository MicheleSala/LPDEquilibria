import numpy as np
from scipy.special import ellipe, ellipk

from coils import Coils


class Equilibrium:
    """
    Class to represent the Equilibrium of the machine on a [r1, r2]x[z1, z2] grid

    Args:
            r1 (float): lower boundary of the computational domain in the r direction
            r2 (float): upper boundary of the computational domain in the r direction
            z1 (float): lower boundary of the computational domain in the z direction
            z2 (float): upper boundary of the computational domain in the z direction
            Nr (int): Number of points in the radial direction
            Nz (int): Number of points in the axial direction
            R (float): numpy NrxNz array
            Z (float): numpy NrxNz array
            psi (float): numpy NrxNz array containing the stream function
            Br (float): numpy NrxNz array containing the r component of the magnetic field
            Bz (float): numpy NrxNz array containing the z component of the magnetic fieldq

    Return:
            Eq (object)
    """

    def __init__(self, r1, r2, z1, z2, Nr, Nz, Nc=10):
        self.__mu0 = 4 * np.pi * 1.0e-07
        self.r1 = r1
        self.r2 = r2
        self.z1 = z1
        self.z2 = z2
        self.Nr = Nr
        self.Nz = Nz
        self.Nc = Nc
        self.Coil = Coils.from_keyboard()
        R, Z = Equilibrium.BuildGrid(self)
        self.R = R
        self.Z = Z
        self.psi = Equilibrium.ComputePsi(self)
        Br, Bz = Equilibrium.ComputeField(self)
        self.Br = Br
        self.Bz = Bz

    def BuildGrid(self):
        """
        Construct cartesian grid in the (R, Z) plane necessary for the computation of the equilibrium
        """
        r1, r2, z1, z2, Nr, Nz = self.r1, self.r2, self.z1, self.z2, self.Nr, self.Nz
        r = np.linspace(r1, r2, Nr)
        z = np.linspace(z1, z2, Nz)
        R, Z = np.meshgrid(r, z)
        return R, Z

    def BuildGridCoils(self, index):
        """
        Compute a cartesian grid on each of the (index) coil with a predefined (Nc=10) number of points
        """
        Rc1, Rc2, Zc1, Zc2, Nc = self.Coil.Rc1, self.Coil.Rc2, self.Coil.Zc1, self.Coil.Zc2, self.Nc
        zc = np.linspace(Zc2[index], Zc1[index], Nc)
        rc = np.linspace(Rc2[index], Rc1[index], Nc)
        Rc, Zc = np.meshgrid(rc, zc)
        return Rc, Zc, rc, zc

    def ComputePsi(self):
        """
        Compute the poloidal flux function using the Green Function method of the Grad-Shafranov Equation
        """
        R, Z, Nr, Nz, Nc, Jc, Ncoil, mu0 = (
            self.R,
            self.Z,
            self.Nr,
            self.Nz,
            self.Nc,
            self.Coil.Jc,
            self.Coil.Ncoils,
            self.__mu0,
        )
        psi_coil = np.zeros((Nz, Nr, Ncoil))
        for ll in range(0, Ncoil):
            G = np.zeros((Nz, Nr, Nc, Nc))
            Rc, Zc, rc, zc = Equilibrium.BuildGridCoils(self, ll)
            for kk in range(0, Nc):
                for mm in range(0, Nc):
                    k2 = (4.0 * R * Rc[kk, mm]) / ((R + Rc[kk, mm]) ** 2 + (Z - Zc[kk, mm]) ** 2)
                    k = np.sqrt(k2)
                    K = ellipk(k2)
                    E = ellipe(k2)
                    G[:, :, kk, mm] = (
                        Jc[ll]
                        * ((mu0 / (2 * np.pi)) * np.sqrt(R * Rc[kk, mm]) / k)
                        * ((2 - k2) * K - 2 * E)
                    )
            psi_coil[:, :, ll] = np.trapezoid(np.trapezoid(G, zc), rc)
        psi = psi_coil.sum(axis=2)
        return psi

    def ComputeField(self):
        """
        Function for the computation of the radial and axial magnetic field components
        """
        Nr, Nz, r2, r1, z2, z1, psi, R = (
            self.Nr,
            self.Nz,
            self.r2,
            self.r1,
            self.z2,
            self.z1,
            self.psi,
            self.R,
        )
        Br = np.zeros((Nz, Nr))
        Bz = np.zeros((Nz, Nr))
        dr = (r2 - r1) / (Nr - 1)
        dz = (z2 - z1) / (Nz - 1)
        # Compute the axial magnetic field
        for ii in range(0, Nz):
            for jj in range(0, Nr):
                if jj == 0:
                    Bz[ii, jj] = 2 * (psi[ii, jj + 1] - psi[ii, jj]) / (dr * dr)
                elif jj == Nr - 1:
                    Bz[ii, jj] = (psi[ii, jj] - psi[ii, jj - 1]) / dr
                    Bz[ii, jj] = Bz[ii, jj] / R[ii, jj]
                else:
                    Bz[ii, jj] = (psi[ii, jj + 1] - psi[ii, jj - 1]) / (2 * dr)
                    Bz[ii, jj] = Bz[ii, jj] / R[ii, jj]
        # Compute the radial magnetic field
        for ii in range(0, Nz):
            for jj in range(0, Nr):
                if ii == 0:
                    Br[ii, jj] = (psi[ii + 1, jj] - psi[ii, jj]) / dz
                elif ii == Nz - 1:
                    Br[ii, jj] = (psi[ii, jj] - psi[ii - 1, jj]) / dz
                else:
                    Br[ii, jj] = (psi[ii + 1, jj] - psi[ii - 1, jj]) / (2 * dz)
                Br[ii, jj] = -Br[ii, jj] / R[ii, jj]
        return Br, Bz
