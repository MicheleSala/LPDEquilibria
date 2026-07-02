import numpy as np


class Coils:
    """
    Class for the representation of the external magnetic field coils

    Args:
            coilName (str): path to the file containing coil data

    Return:
            Jc (array): np.array of length Ncoils containting current density flowing into each coil
    """

    def __init__(self, coilName):
        """
        Initialise the main coil properties starting from a .txt file
        """
        CoilParam = np.loadtxt(coilName)
        self.Ncoils = np.shape(CoilParam)[1]
        self.Rc1 = CoilParam[0, :]
        self.Zc1 = CoilParam[1, :]
        self.Rc2 = CoilParam[2, :]
        self.Zc2 = CoilParam[3, :]
        self.Ic = CoilParam[4, :]
        self.Jc = Coils.ComputeCurrents(self)

    @classmethod
    def from_keyboard(cls):
        coilName = input("Insert coil file name:")
        coilName = "/Users/michele/Desktop/LPDEquilibria/Coils/" + coilName
        Coils.__init__(cls, coilName)
        return cls

    def ComputeCurrents(self):
        """
        Compute the coils current density, assuming that the coil is a square
        """
        Jc = np.zeros(self.Ncoils)
        Jc = self.Ic / ((self.Rc2 - self.Rc1) * (self.Zc2 - self.Zc1))
        return Jc
