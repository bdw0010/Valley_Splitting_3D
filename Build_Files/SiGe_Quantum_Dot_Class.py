"""
    Central class for the 3D SiGe quantum dot simulation
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import parameters as par
import scipy.linalg as linalg
import scipy.sparse as Spar
import scipy.sparse.linalg as SparLinalg
import Lattice_Constant_Funcs as LCF
import SiGe_Quantum_Dot_GeProfile_subClass as SGQDGPSC
np.set_printoptions(linewidth = 500)

class SiGe_Quantum_Dot:

    def __init__(self,nGe_bar,Lx_targ,Ly_targ):

        ### Calculate lattice constants
        self.nGe_bar = nGe_bar # Germanium concentration in the barrier regions
        self.a_par = LCF.calc_relaxed_lat_constant(nGe_bar) # in-plane lattice constant (in Angstroms) set by barrier Ge concentration
        self.b = self.a_par/np.sqrt(2.)                     # in-plane lattice constant (in Angstroms) of square lattice for each atomic layer

        ### Calculate the number of sites in each direction, layer, and total
        self.Nx = int(Lx_targ/self.b) - 1       # number of lattice sites in x-direction for each layer
        self.Ny = int(Ly_targ/self.b) - 1       # number of lattice sites in y-direction for each layer
        self.N_layer = self.Nx * self.Ny        # number of lattice sites in each layer
        self.Lx = (self.Nx + 1) * self.b
        self.Ly = (self.Ny + 1) * self.b

        ### Sub object to handle the generation of Ge profile
        self.Ge_Profile = SGQDGPSC.GeProfile_subObject(self)


    def set_Ge_conc_arr(self,Ge_conc_arr):
        self.Ge_conc_arr = Ge_conc_arr
        self.generate_position_array()
        self.generate_neighbor_array()

    def generate_position_array(self):
        ### Generates the position array that specifies the position of
        ### every atom in the quantum dot simulation
        ###     * Ge_conc_arr specifies the Ge concentration within
        ####      each of the atomic layers along the z-direction

        def z_arr_gen(a_par,Ge_conc_arr):
            ### Generates the z-coordinate array for all of the
            ### atomic layers

            z_arr = np.zeros(Ge_conc_arr.size)
            for m in range(z_arr.size):
                if m == 0:
                    z_arr[m] = 0.
                else:
                    a_perp_m = LCF.calc_m_perp_lat_constant(a_par,Ge_conc_arr[m])
                    a_perp_mM1 = LCF.calc_m_perp_lat_constant(a_par,Ge_conc_arr[m-1])
                    a_perp_m_mM1 = .5 * (a_perp_m + a_perp_mM1)
                    z_arr[m] = z_arr[m-1] + a_perp_m_mM1/4.
            return z_arr

        Ge_conc_arr = self.Ge_conc_arr
        Nx = self.Nx; Ny = self.Ny
        Nz = Ge_conc_arr.size
        self.Nz = Ge_conc_arr.size              # total number of atomic layers in the z-direction
        self.N_sites = self.Nz*self.N_layer     # total number of lattice sites in the whole system

        self.pos_arr = np.zeros((self.N_sites,3)) # position array
        z_arr = z_arr_gen(self.a_par,Ge_conc_arr) # z-coordinate array of all of the atomic layers
        for m in range(Nz):
            zm = z_arr[m]
            idx_m = (Nx*Ny)*m # starting atom index of the current (mth) layer

            ### set lattice shift in xy-plane for the current (mth) atomic layer
            if m % 4 == 0:
                x_sm = -self.Lx/2. - self.b/2.
                y_sm = -self.Ly/2. - self.b/2.
            elif m % 4 == 1:
                x_sm = -self.Lx/2. + self.b/2.
                y_sm = -self.Ly/2. - self.b/2.
            elif m % 4 == 2:
                x_sm = -self.Lx/2. + self.b/2.
                y_sm = -self.Ly/2. + self.b/2.
            elif m % 4 == 3:
                x_sm = -self.Lx/2. - self.b/2.
                y_sm = -self.Ly/2. + self.b/2.

            ### Loop through all atoms in the current (mth) atomic layer
            for j in range(Ny):
                yj = j*self.b + y_sm
                idx_mj = idx_m + Nx*j    # starting atom index of the jth row of the mth layer
                for i in range(Nx):
                    xi = i*self.b + x_sm
                    idx = idx_mj + i     # atom index
                    self.pos_arr[idx,:] = [xi,yj,zm]

    def generate_neighbor_array(self):
        ### Generates the nearest neighbor array which specifies the nearest neighbors
        ### of each atom
        ###     * near_neig_arr[idx,j] = is the index of the jth nearest neighbor of atom idx
        ###         * j = 0,1 are the neighbors above, while j = 2,3 are the neighbors below

        Nx = self.Nx; Ny = self.Ny; Nlay = self.N_layer; Nz = self.Nz
        nna = np.zeros((Nx*Ny*Nz,4),dtype = 'int') # nna stands for nearest_neighbor_array

        ### Loop through all of the atoms
        for m in range(Nz):
            idx_m = m*Nlay
            for j in range(Ny):
                for i in range(Nx):
                    idx = idx_m + Nx*j + i

                    ### Fill in the nearest neighbor array for atom with index idx
                    if m % 4 == 0:
                        if i == 0:
                            nna[idx,0] = idx + Nlay + Nx - 1
                        else:
                            nna[idx,0] = idx + Nlay - 1
                        nna[idx,1] = idx + Nlay
                        if j == 0:
                            nna[idx,2] = idx - Nlay + (Ny - 1)*Nx
                        else:
                            nna[idx,2] = idx - Nlay - Nx
                        nna[idx,3] = idx - Nlay

                    elif m % 4 == 1:
                        if j == 0:
                            nna[idx,0] = idx + Nlay + (Ny - 1)*Nx
                        else:
                            nna[idx,0] = idx + Nlay - Nx
                        nna[idx,1] = idx + Nlay
                        nna[idx,2] = idx - Nlay
                        if i == Nx - 1:
                            nna[idx,3] = idx - Nlay - Nx + 1
                        else:
                            nna[idx,3] = idx - Nlay + 1

                    elif m % 4 == 2:
                        nna[idx,0] = idx + Nlay
                        if i == Nx - 1:
                            nna[idx,1] = idx + Nlay - Nx + 1
                        else:
                            nna[idx,1] = idx + Nlay + 1
                        nna[idx,2] = idx - Nlay
                        if j == Ny - 1:
                            nna[idx,3] = idx - Nlay + (Ny - 1)*Nx
                        else:
                            nna[idx,3] = idx - Nlay + Nx

                    elif m % 4 == 3:
                        nna[idx,0] = idx + Nlay
                        if j == Ny - 1:
                            nna[idx,1] = idx + Nlay - (Ny - 1)*Nx
                        else:
                            nna[idx,1] = idx + Nlay + Nx
                        if i == 0:
                            nna[idx,2] = idx - Nlay + Nx - 1
                        else:
                            nna[idx,2] = idx - Nlay - 1
                        nna[idx,3] = idx - Nlay

        ### Fix the nearest neighbor array for the first and last layers
        for j in range(Ny):
            for i in range(Nx):
                idx = Nx*j + i  # atom index in the first layer
                nna[idx,2] = nna[idx,2] + self.N_sites
                nna[idx,3] = nna[idx,3] + self.N_sites
                idx = Nx*Ny*(Nz - 1) + Nx*j + i  # atom index in the last layer
                nna[idx,0] = nna[idx,0] - self.N_sites
                nna[idx,1] = nna[idx,1] - self.N_sites
        self.near_neig_arr = nna[:,:]



### Testing
if True:

    n_bar = .3
    n_well = 0.
    LX = 40. * 10.
    LY = 40. * 10.
    N_bar = 20
    N_well = 72

    system = SiGe_Quantum_Dot(n_bar,LX,LY)
    Ge_Conc_Arr = system.Ge_Profile.uniform_profile(N_bar,N_well,n_bar,n_well,PLOT = True)
    system.set_Ge_conc_arr(Ge_Conc_Arr)
    print(system.N_sites)
