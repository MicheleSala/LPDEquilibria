import numpy as np
import scipy
from scipy import interpolate
from scipy.special import ellipk, ellipe
import matplotlib
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.interpolate
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif', 'mathtext.fontset': 'stix'})

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
		coilName = input('Insert coil file name:')
		coilName = "../Coils/" + coilName 
		Coils.__init__(cls, coilName)
		return cls
	def ComputeCurrents(self):
		"""
		Compute the coils current density, assuming that the coil is a square
		"""
		Jc = np.zeros(self.Ncoils)
		Jc = self.Ic / ( (self.Rc2 - self.Rc1) * (self.Zc2 - self.Zc1) )
		return Jc


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
	def __init__(self, r1, r2, z1, z2, Nr, Nz, Nc = 10):
		self.__mu0 = 4 * np.pi * 1.0E-07
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
		R, Z, Nr, Nz, Nc, Jc, Ncoil, mu0 = self.R, self.Z, self.Nr, self.Nz, self.Nc, self.Coil.Jc, self.Coil.Ncoils, self.__mu0
		psi_coil = np.zeros((Nz, Nr, Ncoil))
		for ll in range(0, Ncoil):
			G = np.zeros((Nz, Nr, Nc, Nc))
			Rc, Zc, rc, zc = Equilibrium.BuildGridCoils(self, ll)
			for kk in range(0, Nc):
				for mm in range(0, Nc):
					k2 = ( 4.0 * R * Rc[kk,mm] ) / ( (R + Rc[kk,mm]) ** 2 + (Z - Zc[kk,mm]) ** 2)
					k = np.sqrt(k2)
					K = ellipk(k2)
					E = ellipe(k2)
					G[:, :, kk,mm] = Jc[ll] * ((mu0 / ( 2 * np.pi)) * np.sqrt( R * Rc[kk,mm]) / k ) * ( (2 - k2) * K - 2 * E)
			psi_coil[:,:,ll] = np.trapz(np.trapz(G, zc), rc)         
		psi = psi_coil.sum(axis = 2)
		return psi
	def ComputeField(self):
		"""
		Function for the computation of the radial and axial magnetic field components
		""" 
		Nr, Nz, r2, r1, z2, z1, psi, R = self.Nr, self.Nz, self.r2, self.r1, self.z2, self.z1, self.psi, self.R
		Br = np.zeros((Nz, Nr))
		Bz = np.zeros((Nz, Nr))
		dr = (r2 - r1) / (Nr - 1)
		dz = (z2 - z1) / (Nz - 1)
		# Compute the axial magnetic field
		for ii in range(0, Nz):
			for jj in range(0, Nr):
				if jj == 0:
					Bz[ii, jj] = 2 * (psi[ii,jj + 1] - psi[ii, jj]) / (dr*dr)
				elif jj == Nr - 1:
					Bz[ii, jj] = (psi[ii,jj] - psi[ii,jj - 1]) / dr
					Bz[ii, jj] = Bz[ii, jj] / R[ii,jj]
				else:
					Bz[ii,jj] = (psi[ii, jj + 1] - psi[ii, jj - 1]) / (2 * dr)
					Bz[ii,jj] =  Bz[ii, jj] / R[ii,jj]
		# Compute the radial magnetic field
		for ii in range(0, Nz):
			for jj in range(0, Nr):
				if ii == 0:
					Br[ii, jj] = (psi[ii + 1, jj] - psi[ii, jj]) / dz
				elif ii == Nz - 1:
					Br[ii, jj] = (psi[ii, jj] - psi[ii - 1, jj]) / dz
				else:
					Br[ii, jj] = (psi[ii + 1, jj] - psi[ii - 1, jj]) / (2 * dz)
				Br[ii,jj] = - Br[ii,jj] / R[ii,jj]
		return Br, Bz


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
		self.__LowerPsiValue = 1.0E-08
		self.__Dim = 24
		self.Rlim = Rlim
		PsiLim = Mesh.ComputeLastPsi(self)
		Mesh.PsiLim = PsiLim
		self.Mesh = Mesh.ComputeGrid(self)
	def ComputeLastPsi(self):
		"""
		Function for the computation of the last magnetic flux surface
		"""
		psi, R, Z, LowerPsiValue, Ncont, Rlim = self.Eq.psi, self.Eq.R, self.Eq.Z, self.__LowerPsiValue,  self.__Ncont, self.Rlim
		ContourLevels = np.linspace(LowerPsiValue, psi.max(), Ncont)
		Cnt = plt.contour(R, Z,psi, ContourLevels)
		lines = []
		for it in range(0, Ncont):
			for line in Cnt.collections[it].get_paths():
				lines.append(line.vertices)
			r = np.array(lines[it])[:, 0].max() - Rlim
			if r >= 0: 
				print('Found LCFS at: ', ContourLevels[it - 1])                
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
		psi, Bz, R, Z, n, m = self.Eq.psi,self.Eq.Bz, self.Eq.R, self.Eq.Z, self.n, self.m
		zQuery = MESH[:, 3]
		BzC = np.zeros(n * (m + 2))  
		InterpFunc = scipy.interpolate.RectBivariateSpline(Z[:, 0], R[0, :], Bz)        
		for ii in range(0, zQuery.size):
			BzC[ii] = InterpFunc(zQuery[ii], rQuery[ii])
		return BzC       
	def ComputeGrid(self):
		LowerPsiValue, PsiLim, m, n, Z, R, z2 = self.__LowerPsiValue, self.PsiLim, self.m, self.n, self.Eq.Z, self.Eq.R, self.Eq.z2
		ContourLevels = np.linspace(np.sqrt(LowerPsiValue), np.sqrt(PsiLim), m) ** 2 
		zValues = Z[:, 0]
		rValues = R[0, :]
		alpha = 1.8
		overshoot = 1.0E-05
		func = lambda x : z2 * ( ( np.tanh(-x)) / (np.tanh(alpha)) * (1+overshoot)) / (1 + overshoot)
		IndexFunc = np.linspace(-alpha, alpha, 2 * n + 1)
		zQuery = func(IndexFunc)
		InterpPsi = Mesh.InterpolatePsi(self, zValues, rValues, zQuery, ContourLevels)
		crxT = zQuery[0 : zQuery.size - 1 : 2]    
		crxB = zQuery[2 : zQuery.size : 2]
		cryLT = InterpPsi[0 : InterpPsi.shape[0] - 1: 2, 0 : InterpPsi.shape[1] - 1].T.reshape(1, n * (m - 1))
		cryRT = InterpPsi[0 : InterpPsi.shape[0] - 1: 2, 1 : InterpPsi.shape[1]].T.reshape(1, n * (m - 1))
		cryLB = InterpPsi[2 : InterpPsi.shape[0] : 2, 0 : InterpPsi.shape[1] - 1].T.reshape(1, n * (m - 1))
		cryRB = InterpPsi[2 : InterpPsi.shape[0] : 2, 1 : InterpPsi.shape[1]].T.reshape(1, n * (m - 1))
		cryBottomGuard = np.zeros((1, n))
		cryUpperGuard = Mesh.InterpolatePsi(self, zValues, rValues, zQuery, np.array([1.0E-12]))
		cryUpperGuardLT = cryUpperGuard[0 : cryUpperGuard.size - 1: 2]
		cryUpperGuardLB = cryUpperGuard[2 : cryUpperGuard.size : 2]       
		cryLT = np.append(np.append(cryBottomGuard, cryUpperGuardLT), cryLT)
		cryLB = np.append(np.append(cryBottomGuard, cryUpperGuardLB), cryLB)
		cryRT = np.append(np.append(cryUpperGuardLT, cryLT[2 * n: 3 * n]), cryRT)
		cryRB = np.append(np.append(cryUpperGuardLB, cryLB[2 * n: 3 * n]), cryRB) 
		cryLB = np.append(cryLB, cryRB[cryRB.size - n: cryRB.size])
		cryLT = np.append(cryLT, cryRT[cryRB.size - n: cryRB.size])
		cryRT = np.append(cryRT, cryRT[cryRB.size - n: cryRB.size] + 1.0E-05)
		cryRB = np.append(cryRB, cryRB[cryRB.size - n: cryRB.size] + 1.0E-05)
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
		MESH[:, 12] = BzC.reshape(n * (m  + 2))
		MESH[:, 13] = 0
		return MESH 
	def PlotMesh(self):
		Mesh = self.Mesh
		it = 0       
		Mesh = np.delete(Mesh, (1), axis = 1)
		Mesh = np.delete(Mesh, (0), axis = 1)
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
			ax.plot(x[ii * self.n * 5 + 1 : Points], y[ii * self.n * 5 + 1 : Points], 'black', linewidth = 0.5)
			Points = Points + self.n * 5
		ax.set_aspect(aspect=0.3)
		ax.set_xlabel('R [m]')
		ax.set_ylabel('Z [m]')
		plt.show()
	def WriteMesh(self): 
		Mesh = self.Mesh
		Name = input('Insert Mesh name: ') + '.ASCII'    
		pathName = 'meshes/' 
		np.savetxt(pathName + Name, Mesh, '%.8E')
		print('Saved Mesh in file: ' + Name)        

class PlotEquilibrium: 
	""" 
	Class for plotting the Equilibrium object
	"""
	def __init__(self, Eq, Ncont = 100, Dim = 24):
		self.Equ = Eq
		self.Ncont = Ncont
		self.Dim = Dim
	def PlotPsiContour(self):
		"""
		Contour plot of psi
		"""
		R, Z, psi, Ncont = self.Equ.R, self.Equ.Z, self.Equ.psi, self.Ncont
		fig, ax = plt.subplots()
		cnt = ax.contour(R, Z, psi, Ncont, cmap=matplotlib.cm.RdGy, vmin=abs(self.Equ.psi).min(), vmax=abs(self.Equ.psi).max())
		ax.set_xlabel('R [m]')
		ax.set_ylabel('Z [m]')
		cb = fig.colorbar(cnt, ax=ax)
		plt.show()
	def PlotBContour(self):
		"""
		Contour plot of B
		""" 
		R, Z, Br, Bz, Ncont = self.Equ.R, self.Equ.Z, self.Equ.Br, self.Equ.Bz, self.Ncont
		B = np.sqrt(self.Equ.Br ** 2 + self.Equ.Bz ** 2)
		fig, ax = plt.subplots()
		cnt = ax.contourf(R, Z, B, Ncont, cmap=matplotlib.cm.RdGy, vmin=abs(B).min(), vmax=abs(B).max())
		ax.set_xlabel('R [m]')
		ax.set_ylabel('Z [m]')
		cb = fig.colorbar(cnt, ax=ax)
		plt.show()
	def PlotPsiBContour(self): 
		"""
		Plot psi and B contour on the same figure
		"""
		R, Z, Br, Bz, psi, Ncont = self.Equ.R, self.Equ.Z, self.Equ.Br, self.Equ.Bz, self.Equ.psi, self.Ncont
		B = np.sqrt(self.Equ.Br ** 2 + self.Equ.Bz ** 2)
		fig, ax = plt.subplots(nrows = 2, ncols = 1)
		cntPsi = ax[0].contour(R[0, :], Z[:, 0], psi, Ncont, cmap=matplotlib.cm.RdGy,vmin=abs(psi).min(), vmax=abs(psi).max())
		cb = fig.colorbar(cntPsi, ax = ax[0])
		ax[0].set_xlabel('R [m]')
		ax[0].set_ylabel('Z [m]')
		cntB = ax[1].contourf(R, Z, B, Ncont, cmap=matplotlib.cm.RdGy,vmin=abs(B).min(), vmax=abs(B).max())
		cb = fig.colorbar(cntB, ax = ax[1])
		ax[1].set_xlabel('R [m]')
		ax[1].set_ylabel('Z [m]')
		plt.show()

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
		Br, Bz, e, me, mi, neFit, Eq = self.Eq.Br, self.Eq.Bz, self.__e, self.__me, self.__mi, self.neFit, self.Eq
		B = np.sqrt(Br ** 2 + Bz ** 2)
		OmegaCE = e * B / (2 * np.pi * me)
		OmegaCI = e * B / (2 * np.pi * mi)  
		#OmegaPE = 8.98 * np.sqrt(neFit(Eq.R, Eq.Z))
		OmegaPE = 8.98 * np.sqrt(interpolate.splev(Eq.R, neFit))
		plt.contourf(Eq.R[0, :], Eq.Z[:, 0], interpolate.splev(Eq.R, neFit))
		plt.colorbar()
		plt.show()
		OmegaPI = 1 / (2 * np.pi) * np.sqrt(me / mi) * OmegaPE
		OmegaUH = np.sqrt(OmegaPE ** 2 + OmegaCE ** 2)
		OmegaLH = np.sqrt(OmegaCE * OmegaCI * (1 + OmegaCI ** 2 / OmegaPI ** 2) / (1 + OmegaCE ** 2 / OmegaPE ** 2))
		return OmegaCE, OmegaUH, OmegaLH, OmegaPE
	def ComputeCutOff(self):
		OmegaCE, OmegaPE = self.OmegaCE, self.OmegaPE
		OmegaC1 = 0.5 * (np.sqrt(4. * OmegaPE ** 2 + OmegaCE ** 2) - OmegaCE)
		OmegaC2 = 0.5 * (np.sqrt(4. * OmegaPE ** 2 + OmegaCE ** 2) + OmegaCE)
		return OmegaC1, OmegaC2  
	def LocateAxial(self): 
		"""Method for the computation of the axial location of the resonace"""
		OmegaInj, Br, Bz, psi, R, Z, e, me, mi, OmegaCE = self.OmegaInj, self.Eq.Br, self.Eq.Bz, self.Eq.psi, self.Eq.R, self.Eq.Z, self.__e, self.__me, self.__mi, self.OmegaCE
		cs = plt.contour(R[0, :], Z[:, 0], OmegaCE, [OmegaInj])
		lines = []
		for line in cs.collections[0].get_paths():
			lines.append(line.vertices)
		zpos = np.empty(len(lines))            
		for ii in range(0, len(lines)):
			zpos[ii] = np.array(lines[ii]) [:, 1].min()           
		return zpos
	def PlotResonances(self): 
		"""Method for plotting of the resonances"""
		OmegaInj, OmegaCE, OmegaUH, OmegaLH, R, Z = self.OmegaInj, self.OmegaCE, self.OmegaUH, self.OmegaLH, self.Eq.R, self.Eq.Z
		fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 10))
		ax[0].contour(R[0, :], Z[:, 0], OmegaCE, [OmegaInj])
		ax[0].set_title('$\Omega_{CE}$')  
		ax[0].set_xlabel('R [m]')
		ax[0].set_ylabel('Z [m]') 
		ax[1].contour(R[0, :], Z[:, 0], OmegaUH, [OmegaInj])
		ax[1].set_title('$\Omega_{UH}$')
		ax[1].set_xlabel('R [m]')
		ax[1].set_ylabel('Z [m]')
		ax[2].contour(R[0, :], Z[:, 0], OmegaLH, [OmegaInj])
		ax[2].set_title('$\Omega_{LH}$')
		ax[2].set_xlabel('R [m]')
		ax[2].set_ylabel('Z [m]')
		plt.show()
		plt.contourf(R[0, :], Z[:, 0], OmegaUH)
		plt.colorbar()
		fig.tight_layout()
	def PlotCutOff(self): 
		"""Method for plotting of the cutoffs"""
		OmegaInj, OmegaC1, OmegaC2, R, Z= self.OmegaInj, self.OmegaC1, self.OmegaC2, self.Eq.R, self.Eq.Z
		fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 10))
		ax[0].contour(R[0, :], Z[:, 0], OmegaC1, [OmegaInj])
		ax[0].set_title('$\Omega_{C1}$')  
		ax[0].set_xlabel('R [m]')
		ax[0].set_ylabel('Z [m]') 
		ax[1].contour(R[0, :], Z[:, 0], OmegaC2, [OmegaInj])
		ax[1].set_title('$\Omega_{C2}$')
		ax[1].set_xlabel('R [m]')
		ax[1].set_ylabel('Z [m]')
		fig.tight_layout()
		plt.show()
		plt.contourf(R[0, :], Z[:, 0], OmegaC2)
		plt.colorbar()
	def PlotResonancesMesh(self): 
		"""Method for plotting of the resonances on the physical mesh"""       
		OmegaCE, OmegaUH, OmegaLH, R, Z, Grid, n, m, OmegaInj = self.OmegaCE, self.OmegaUH, self.OmegaLH, self.Eq.R, self.Eq.Z, self.Mesh.Mesh, self.Mesh.n, self.Mesh.m, self.OmegaInj
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
		Grid = np.delete(Grid, (1), axis = 1)
		Grid = np.delete(Grid, (0), axis = 1)
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
		fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 10))
		ax[0].contour(Rc, Zc, OmegaCEGrid, [OmegaInj]) 
		ax[0].set_title('$\Omega_{CE}$')  
		ax[0].set_xlabel('R [m]')
		ax[0].set_ylabel('Z [m]')         
		for ii in range(0, m + 3):
			ax[0].plot(x[ii * n * 5 + 1 : Points], y[ii * n * 5 + 1 : Points], 'black', linewidth = 0.5)
			Points = Points + n * 5
		Points = n * 5  
		ax[1].contour(Rc, Zc, OmegaUHGrid, [OmegaInj]) 
		ax[1].set_title('$\Omega_{UH}$')  
		ax[1].set_xlabel('R [m]')
		ax[1].set_ylabel('Z [m]')
		for ii in range(0, m + 3):
			ax[1].plot(x[ii * n * 5 + 1 : Points], y[ii * n * 5 + 1 : Points], 'black', linewidth = 0.5)
			Points = Points + n * 5
		Points = n * 5  
		ax[2].contour(Rc, Zc, OmegaLHGrid, [OmegaInj]) 
		ax[2].set_title('$\Omega_{LH}$')  
		ax[2].set_xlabel('R [m]')
		ax[2].set_ylabel('Z [m]')
		for ii in range(0, m + 3):
			ax[2].plot(x[ii * n * 5 + 1 : Points], y[ii * n * 5 + 1 : Points], 'black', linewidth = 0.5)
			Points = Points + n * 5
		Points = n * 5  
		plt.show()
		fig.tight_layout()

class Sources:
	"""Class for the definition of the source input """
	def __init__(self, RadialDensity, AxialDensity, Mesh, Resonances, sigma):
		self.Grid = Mesh
		self.Res = Resonances
		self.zmin = Resonances.ZResPosition - sigma
		self.zmax = Resonances.ZResPosition + sigma
		self.RadialDensity = RadialDensity
		self.AxialDensity = AxialDensity 
		self.RadialPoints = Sources.ComputeRadialPoints(self)
		self.AxialPoints= Sources.ComputeAxialPoints(self)
	def ComputeRadialPoints(self):
		AxialResPosition, Mesh, n, zmin, zmax= self.Res.ZResPosition, self.Grid.Mesh, self.Grid.n, self.zmin, self.zmax
		Shape = np.shape(Mesh)
		Toll = AxialResPosition * 1.1
		zMin = AxialResPosition - Toll
		zMax = AxialResPosition + Toll 
		RcentrePoints = Mesh[0 : Shape[0], 2]
		ZcentrePoints = Mesh[0 : Shape[0], 3]
		ZonAxis = ZcentrePoints[0 : n]
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
		RadialPoints, AxialPoints = self.RadialPoints, self.AxialPoints
		RadialValues = self.RadialDensity(RadialPoints)
		AxialValues = self.AxialDensity(AxialPoints)
		l = np.size(RadialPoints)



