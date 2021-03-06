"""
    Subobject class to handle the Hamiltonian generation for the quantum dot
"""

import numpy as np
import parameters as par
import scipy.sparse as Spar
import scipy.sparse.linalg as SparLinalg
np.set_printoptions(linewidth = 500)

class SiGe_Quantum_Dot_Ham:

    def __init__(self,QD_obj):
        self.QD_obj = QD_obj # parent instance of SiGe_Quantum_Dot

    def unstrained_Si_Ham_gen(self):
        ### Generate the Hamiltonian of an unstrained Si system
        ###     * This function ignores any Ge atoms or strain in the system
        ###     * Does not include any potential term.
        ###         * A potential needs to be handled separately
        ###     * Excludes spin-orbit coupling
        ###     * Uses a sp3s* basis to reduce the size of the resulting matrix
        ###     * Primarily used for testing the tight binding code in a simplified
        ###       system
        pass

    def intraAtomic_Ham_gen(self):
        ### Generates the onsite component (intra-atomic) of the Hamiltonian
        ### in the absence of any potential
        ###     * Excluded spin-orbit coupling
        ###     * Using the sp3d5s* basis with orbital ordering {s,s*,px,py,pz,yz,xz,xy,x2my2,z2}

        ### parameters
        if True:
            a_Si = 5.431; a_Ge = 5.6563
            dSi_o = np.sqrt(3.*a_Si**2/16.) # unstrained bond distance of Si
            dGe_o = np.sqrt(3.*a_Ge**2/16.) # unstrained bond distance of Si

            ### Bare orbital energies
            Es_Si = -2.55247
            Ep_Si = 4.48593
            Ed_Si = 14.01053
            Es2_Si = 23.44607

            E_offset = 0.68
            Es_Ge = -4.08253 + E_offset
            Ep_Ge = 4.63470 + E_offset
            Ed_Ge = 12.19526 + E_offset
            Es2_Ge = 23.20167 + E_offset

            ### Strain parameters
            alp_s_Si = -0.13357
            alp_p_Si = -0.18953
            alp_d_Si = -0.89046
            alp_s2_Si = -0.24373
            beta_p0_Si = 1.13646
            beta_p1_Si = -2.76257
            beta_pd0_Si = -0.13011
            beta_pd1_Si = -3.28537
            beta_d0_Si = 3.59603
            beta_sp0_Si = 1.97665
            beta_s2p0_Si = -2.18403
            beta_sd0_Si = 3.06840
            beta_s2d0_Si = -4.95860

            alp_s_Ge = -0.33252
            alp_p_Ge = -0.43824
            alp_d_Ge = -0.90486
            alp_s2_Ge = -0.52062
            beta_p0_Ge = 1.01233
            beta_p1_Ge = -2.53951
            beta_pd0_Ge = -0.22597
            beta_pd1_Ge = -3.77180
            beta_d0_Ge = 1.99217
            beta_sp0_Ge = 1.27627
            beta_s2p0_Ge = -2.02374
            beta_sd0_Ge = 2.38822
            beta_s2d0_Ge = -4.73191

        row = []; col = []; data = []
        ### Loop through the layers in the system
        for m in range(self.QD_obj.Nz):
            idx_m = m*self.QD_obj.N_layer
            cosines = self.QD_obj.cosines_arr[idx_m,:,:]

            ### Calculate the hydrostatic strain
            hs_Si_m = 0.75*(np.sum(cosines[:,3])/dSi_o - 4)
            hs_Ge_m = 0.75*(np.sum(cosines[:,3])/dGe_o - 4)
            #print(self.QD_obj.Nz - m,hs_Si_m,hs_Ge_m)

            ### Calculate the onsite energy of s and s* orbitals
            Es_Si_m = Es_Si + alp_s_Si*hs_Si_m
            Es2_Si_m = Es2_Si + alp_s2_Si*hs_Si_m
            Es_Ge_m = Es_Ge + alp_s_Ge*hs_Ge_m
            Es2_Ge_m = Es2_Ge + alp_s2_Ge*hs_Ge_m

            ### calculate the onsite energy of p orbitals
            Ep_Si_m = np.ones(3)*(Ep_Si + alp_p_Si*hs_Si_m)
            Ep_Ge_m = np.ones(3)*(Ep_Ge + alp_p_Ge*hs_Ge_m)
            for j in range(4):
                dj_ratio_Si = (cosines[j,3] - dSi_o)/dSi_o
                dj_ratio_Ge = (cosines[j,3] - dGe_o)/dGe_o
                beta_Si_j = beta_p0_Si + beta_p1_Si*dj_ratio_Si
                beta_Ge_j = beta_p0_Ge + beta_p1_Ge*dj_ratio_Ge
                for l in range(3):
                    Ep_Si_m[l] += beta_Si_j*(cosines[j,l]**2 - 1./3.)
                    Ep_Ge_m[l] += beta_Ge_j*(cosines[j,l]**2 - 1./3.)

            ### calculate the ontie terms between d orbitals
            Ed_Si_m = np.ones(5)*(Ed_Si + alp_d_Si*hs_Si_m)
            Ed_Ge_m = np.ones(5)*(Ed_Ge + alp_d_Ge*hs_Ge_m)
            Ed_Si_m[0] += beta_d0_Si*(np.sum(np.square(cosines[:,0])) - 4./3.)
            Ed_Si_m[1] += beta_d0_Si*(np.sum(np.square(cosines[:,1])) - 4./3.)
            Ed_Si_m[2] += beta_d0_Si*(np.sum(np.square(cosines[:,2])) - 4./3.)
            Ed_Si_m[3] += beta_d0_Si*(np.sum(np.square(cosines[:,2])) - 4./3.)
            Ed_Si_m[4] -= beta_d0_Si*(np.sum(np.square(cosines[:,2])) - 4./3.)
            Ed_Ge_m[0] += beta_d0_Ge*(np.sum(np.square(cosines[:,0])) - 4./3.)
            Ed_Ge_m[1] += beta_d0_Ge*(np.sum(np.square(cosines[:,1])) - 4./3.)
            Ed_Ge_m[2] += beta_d0_Ge*(np.sum(np.square(cosines[:,2])) - 4./3.)
            Ed_Ge_m[3] += beta_d0_Ge*(np.sum(np.square(cosines[:,2])) - 4./3.)
            Ed_Ge_m[4] -= beta_d0_Ge*(np.sum(np.square(cosines[:,2])) - 4./3.)
            E_x2my2_z2_Si_m = np.sqrt(1./3.)*beta_d0_Si*np.sum(np.square(cosines[:,0]) - np.square(cosines[:,1]))
            E_x2my2_z2_Ge_m = np.sqrt(1./3.)*beta_d0_Ge*np.sum(np.square(cosines[:,0]) - np.square(cosines[:,1]))

            ### calculate the onsite coupling between s and pz orbitals
            E_s_pz_Si_m = beta_sp0_Si*np.sum(cosines[:,2])
            E_s_pz_Ge_m = beta_sp0_Ge*np.sum(cosines[:,2])
            E_s2_pz_Si_m = beta_s2p0_Si*np.sum(cosines[:,2])
            E_s2_pz_Ge_m = beta_s2p0_Ge*np.sum(cosines[:,2])

            ### calculate the onsite coupling between s and d orbitals
            E_s_x2my2_Si_m = 0.5 * beta_sd0_Si * np.sum(np.square(cosines[:,0]) - np.square(cosines[:,1]))
            E_s_x2my2_Ge_m = 0.5 * beta_sd0_Ge * np.sum(np.square(cosines[:,0]) - np.square(cosines[:,1]))
            E_s_z2_Si_m = (1./(2*np.sqrt(3))) * beta_sd0_Si * (np.sum(np.square(cosines[:,2])) - 4)
            E_s_z2_Ge_m = (1./(2*np.sqrt(3))) * beta_sd0_Ge * (np.sum(np.square(cosines[:,2])) - 4)
            E_s2_x2my2_Si_m = 0.5 * beta_s2d0_Si * np.sum(np.square(cosines[:,0]) - np.square(cosines[:,1]))
            E_s2_x2my2_Ge_m = 0.5 * beta_s2d0_Ge * np.sum(np.square(cosines[:,0]) - np.square(cosines[:,1]))
            E_s2_z2_Si_m = (1./(2*np.sqrt(3))) * beta_s2d0_Si * (np.sum(np.square(cosines[:,2])) - 4)
            E_s2_z2_Ge_m = (1./(2*np.sqrt(3))) * beta_s2d0_Ge * (np.sum(np.square(cosines[:,2])) - 4)

            ### calcualte the onsite coupling between p and d orbitals
            E_px_xz_Si_m = beta_pd0_Si * np.sum(cosines[:,2])
            for j in range(4):
                dj_ratio_Si = (cosines[j,3] - dSi_o)/dSi_o
                E_px_xz_Si_m += beta_pd1_Si * dj_ratio_Si * cosines[j,2]
            E_py_yz_Si_m = 1.*E_px_xz_Si_m
            E_pz_z2_Si_m = (2./np.sqrt(3.))*E_px_xz_Si_m
            E_px_xz_Ge_m = beta_pd0_Ge * np.sum(cosines[:,2])
            for j in range(4):
                dj_ratio_Ge = (cosines[j,3] - dGe_o)/dGe_o
                E_px_xz_Ge_m += beta_pd1_Ge * dj_ratio_Ge * cosines[j,2]
            E_py_yz_Ge_m = 1.*E_px_xz_Ge_m
            E_pz_z2_Ge_m = (2./np.sqrt(3.))*E_px_xz_Ge_m

            print(m,E_x2my2_z2_Si_m)

            ### Loop thorugh all of the atoms in layer m
            N_orb = 10 # number of orbitals per atom
            for i in range(self.QD_obj.N_layer):
                idx_mi = idx_m + i          # atom index
                idx_orb = idx_mi * N_orb    # starting orbital index
                if self.QD_obj.atom_type_arr[idx_mi] == 0: # Si atom

                    ### Diagonal terms
                    row.append(idx_orb + 0); col.append(idx_orb + 0); data.append(Es_Si_m)
                    row.append(idx_orb + 1); col.append(idx_orb + 1); data.append(Es2_Si_m)
                    row.append(idx_orb + 2); col.append(idx_orb + 2); data.append(Ep_Si_m[0])
                    row.append(idx_orb + 3); col.append(idx_orb + 3); data.append(Ep_Si_m[1])
                    row.append(idx_orb + 4); col.append(idx_orb + 4); data.append(Ep_Si_m[2])
                    row.append(idx_orb + 5); col.append(idx_orb + 5); data.append(Ed_Si_m[0])
                    row.append(idx_orb + 6); col.append(idx_orb + 6); data.append(Ed_Si_m[1])
                    row.append(idx_orb + 7); col.append(idx_orb + 7); data.append(Ed_Si_m[2])
                    row.append(idx_orb + 8); col.append(idx_orb + 8); data.append(Ed_Si_m[3])
                    row.append(idx_orb + 9); col.append(idx_orb + 9); data.append(Ed_Si_m[4])

                    ### s-p terms
                    row.append(idx_orb + 0); col.append(idx_orb + 4); data.append(E_s_pz_Si_m)
                    col.append(idx_orb + 0); row.append(idx_orb + 4); data.append(E_s_pz_Si_m)
                    row.append(idx_orb + 1); col.append(idx_orb + 4); data.append(E_s2_pz_Si_m)
                    col.append(idx_orb + 1); row.append(idx_orb + 4); data.append(E_s2_pz_Si_m)

                    ### s-d terms
                    row.append(idx_orb + 0); col.append(idx_orb + 8); data.append(E_s_x2my2_Si_m)
                    col.append(idx_orb + 0); row.append(idx_orb + 8); data.append(E_s_x2my2_Si_m)
                    row.append(idx_orb + 1); col.append(idx_orb + 8); data.append(E_s2_x2my2_Si_m)
                    col.append(idx_orb + 1); row.append(idx_orb + 8); data.append(E_s2_x2my2_Si_m)
                    row.append(idx_orb + 0); col.append(idx_orb + 9); data.append(E_s_z2_Si_m)
                    col.append(idx_orb + 0); row.append(idx_orb + 9); data.append(E_s_z2_Si_m)
                    row.append(idx_orb + 1); col.append(idx_orb + 9); data.append(E_s2_z2_Si_m)
                    col.append(idx_orb + 1); row.append(idx_orb + 9); data.append(E_s2_z2_Si_m)

                    ### p-d terms
                    row.append(idx_orb + 2); col.append(idx_orb + 6); data.append(E_px_xz_Si_m)
                    col.append(idx_orb + 2); row.append(idx_orb + 6); data.append(E_px_xz_Si_m)
                    row.append(idx_orb + 3); col.append(idx_orb + 5); data.append(E_py_yz_Si_m)
                    col.append(idx_orb + 3); row.append(idx_orb + 5); data.append(E_py_yz_Si_m)
                    row.append(idx_orb + 4); col.append(idx_orb + 9); data.append(E_pz_z2_Si_m)
                    col.append(idx_orb + 4); row.append(idx_orb + 9); data.append(E_pz_z2_Si_m)

                    ### d-d terms
                    row.append(idx_orb + 8); col.append(idx_orb + 9); data.append(E_x2my2_z2_Si_m)
                    col.append(idx_orb + 8); row.append(idx_orb + 9); data.append(E_x2my2_z2_Si_m)

                elif self.QD_obj.atom_type_arr[idx_mi] == 1: # Ge atom

                    ### Diagonal terms
                    row.append(idx_orb + 0); col.append(idx_orb + 0); data.append(Es_Ge_m)
                    row.append(idx_orb + 1); col.append(idx_orb + 1); data.append(Es2_Ge_m)
                    row.append(idx_orb + 2); col.append(idx_orb + 2); data.append(Ep_Ge_m[0])
                    row.append(idx_orb + 3); col.append(idx_orb + 3); data.append(Ep_Ge_m[1])
                    row.append(idx_orb + 4); col.append(idx_orb + 4); data.append(Ep_Ge_m[2])
                    row.append(idx_orb + 5); col.append(idx_orb + 5); data.append(Ed_Ge_m[0])
                    row.append(idx_orb + 6); col.append(idx_orb + 6); data.append(Ed_Ge_m[1])
                    row.append(idx_orb + 7); col.append(idx_orb + 7); data.append(Ed_Ge_m[2])
                    row.append(idx_orb + 8); col.append(idx_orb + 8); data.append(Ed_Ge_m[3])
                    row.append(idx_orb + 9); col.append(idx_orb + 9); data.append(Ed_Ge_m[4])

                    ### s-p terms
                    row.append(idx_orb + 0); col.append(idx_orb + 4); data.append(E_s_pz_Ge_m)
                    col.append(idx_orb + 0); row.append(idx_orb + 4); data.append(E_s_pz_Ge_m)
                    row.append(idx_orb + 1); col.append(idx_orb + 4); data.append(E_s2_pz_Ge_m)
                    col.append(idx_orb + 1); row.append(idx_orb + 4); data.append(E_s2_pz_Ge_m)

                    ### s-d terms
                    row.append(idx_orb + 0); col.append(idx_orb + 8); data.append(E_s_x2my2_Ge_m)
                    col.append(idx_orb + 0); row.append(idx_orb + 8); data.append(E_s_x2my2_Ge_m)
                    row.append(idx_orb + 1); col.append(idx_orb + 8); data.append(E_s2_x2my2_Ge_m)
                    col.append(idx_orb + 1); row.append(idx_orb + 8); data.append(E_s2_x2my2_Ge_m)
                    row.append(idx_orb + 0); col.append(idx_orb + 9); data.append(E_s_z2_Ge_m)
                    col.append(idx_orb + 0); row.append(idx_orb + 9); data.append(E_s_z2_Ge_m)
                    row.append(idx_orb + 1); col.append(idx_orb + 9); data.append(E_s2_z2_Ge_m)
                    col.append(idx_orb + 1); row.append(idx_orb + 9); data.append(E_s2_z2_Ge_m)

                    ### p-d terms
                    row.append(idx_orb + 2); col.append(idx_orb + 6); data.append(E_px_xz_Ge_m)
                    col.append(idx_orb + 2); row.append(idx_orb + 6); data.append(E_px_xz_Ge_m)
                    row.append(idx_orb + 3); col.append(idx_orb + 5); data.append(E_py_yz_Ge_m)
                    col.append(idx_orb + 3); row.append(idx_orb + 5); data.append(E_py_yz_Ge_m)
                    row.append(idx_orb + 4); col.append(idx_orb + 9); data.append(E_pz_z2_Ge_m)
                    col.append(idx_orb + 4); row.append(idx_orb + 9); data.append(E_pz_z2_Ge_m)

                    ### d-d terms
                    row.append(idx_orb + 8); col.append(idx_orb + 9); data.append(E_x2my2_z2_Ge_m)
                    col.append(idx_orb + 8); row.append(idx_orb + 9); data.append(E_x2my2_z2_Ge_m)

        Ham_onsite = Spar.csc_matrix((data,(row,col)), shape=(N_orb*self.QD_obj.N_sites, N_orb*self.QD_obj.N_sites))
        return Ham_onsite

def calc_interatomic_mtx_element_Si_Si(??,??,l,m,n,d):
    ### Calculates the interatomic matrix element between the
    ### orbital ?? on the reference Si atom and the ?? orbital on
    ### the target Si atom
    ###     * l,m,n are the directional cosines of the vector
    ###       going from the reference to the target atom
    ###     * d is the distance between atoms

    d_Si = 5.431 * np.sqrt(3.)/4.
    d_ratio = (d_Si/d) # ratio of unstrained and strained inter-atomic distance

    ### Unstrained band parameters
    Es = -2.55247 # + E_offset
    Ep = 4.48593 # + E_offset
    Ed = 14.01053 # + E_offset
    Es2 = 23.44607 # + E_offset

    Vss?? = -1.86600
    Vsp?? = 2.91067
    Vsd?? = -2.23992
    Vss2?? = -1.39107

    Vpp?? = 4.08481
    Vpp?? = -1.49207
    Vpd?? = -1.66657
    Vpd?? = 2.39936

    Vdd?? = -1.82945
    Vdd?? = 3.08177
    Vdd?? = -1.56676

    Vs2p?? = 3.06822
    Vs2d?? = -0.77711
    Vs2s2?? = -4.51331

    Delta_SO = 3*0.01851

    ### Deformation exponentials
    nss?? = 3.56701
    nss2?? = 1.51967
    nsp?? = 2.03530
    nsd?? = 2.14811

    ns2s2?? = 0.64401
    ns2p?? = 1.46652
    ns2d?? = 1.79667

    npp?? = 2.01907
    npp?? = 2.87276
    npd?? = 1.00446
    npd?? = 1.78029

    ndd?? = 1.73865
    ndd?? = 1.80442
    ndd?? = 2.54691
    #bd = 0.443

    ### Calculate the renormalized matrix elements
    Vss?? = Vss?? * (d_ratio**nss??)
    Vss2?? = Vss2?? * (d_ratio**nss2??)
    Vs2s?? = Vss2?? * (d_ratio**nss2??)
    Vsp?? = Vsp?? * (d_ratio**nsp??)
    Vps?? = Vsp?? * (d_ratio**nsp??)
    Vsd?? = Vsd?? * (d_ratio**nsd??)
    Vds?? = Vsd?? * (d_ratio**nsd??)

    Vs2s2?? = Vs2s2?? * (d_ratio**ns2s2??)
    Vs2p?? = Vs2p?? * (d_ratio**ns2p??)
    Vps2?? = Vs2p?? * (d_ratio**ns2p??)
    Vs2d?? = Vs2d?? * (d_ratio**ns2d??)
    Vds2?? = Vs2d?? * (d_ratio**ns2d??)

    Vpp?? = Vpp?? * (d_ratio**npp??)
    Vpp?? = Vpp?? * (d_ratio**npp??)
    Vpd?? = Vpd?? * (d_ratio**npd??)
    Vdp?? = Vpd?? * (d_ratio**npd??)
    Vpd?? = Vpd?? * (d_ratio**npd??)
    Vdp?? = Vpd?? * (d_ratio**npd??)

    Vdd?? = Vdd?? * (d_ratio**ndd??)
    Vdd?? = Vdd?? * (d_ratio**ndd??)
    Vdd?? = Vdd?? * (d_ratio**ndd??)


    if (?? == 's'):
        if (?? == 's'):
            mtx_elem = Vss??
        elif (?? == 'px'):
            mtx_elem = l*Vsp??
        elif (?? == 'py'):
            mtx_elem = m*Vsp??
        elif (?? == 'pz'):
            mtx_elem = n*Vsp??
        elif (?? == 'xy'):
            mtx_elem = np.sqrt(3)*l*m*Vsd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3)*m*n*Vsd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3)*l*n*Vsd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vsd??
        elif ?? == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vsd??
        elif ?? == 's2':
            mtx_elem = Vss2??
        else:
            print("Error! s")
            sys.exit()

    elif (?? == 'px'):
        if ?? == 's':
            mtx_elem = -l*Vps??
        elif ?? == 'px':
            mtx_elem = l**2 * Vpp?? + (1 - l**2) * Vpp??
        elif ?? == 'py':
            mtx_elem = l*m*Vpp?? - l*m*Vpp??
        elif ?? == 'pz':
            mtx_elem = l*n*Vpp?? - l*n*Vpp??
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3) * l**2 * m * Vpd?? + m*(1 - 2*l**2)*Vpd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3) * l*m*n * Vpd?? - 2*l*m*n*Vpd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3) * l**2 * n * Vpd?? + n*(1 - 2*l**2)*Vpd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*l*(l**2 - m**2) * Vpd?? + l*(1 - l**2 + m**2) * Vpd??
        elif ?? == 'z2mr2':
            mtx_elem = l*(n**2 - (l**2 + m**2)/2.)*Vpd?? - np.sqrt(3) * l * n**2 * Vpd??
        elif ?? == 's2':
            mtx_elem = -l*Vps2??
        else:
            print("Error! px")
            sys.exit()

    elif (?? == 'py'):
        if ?? == 's':
            mtx_elem = -m*Vps??
        elif ?? == 'px':
            mtx_elem = l*m*Vpp?? - l*m*Vpp??
        elif ?? == 'py':
            mtx_elem = m**2 * Vpp?? + (1. - m**2)*Vpp??
        elif ?? == 'pz':
            mtx_elem = m*n*Vpp?? - m*n*Vpp??
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3) * m**2 * l * Vpd?? + l*(1-2*m**2)*Vpd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3) * m**2 * n * Vpd?? + n*(1 - 2*m**2)*Vpd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3) * l*m*n * Vpd?? - 2*l*m*n*Vpd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*m*(l**2 - m**2)*Vpd?? - m*(1 +  l**2 - m**2)*Vpd??
        elif ?? == 'z2mr2':
            mtx_elem = m*(n**2 - (l**2 + m**2)/2.)*Vpd?? - np.sqrt(3)*m*n**2*Vpd??
        elif ?? == 's2':
            mtx_elem = -m*Vps2??
        else:
            print("Error! py")
            sys.exit()

    elif (?? == 'pz'):
        if ?? == 's':
            mtx_elem = -n*Vps??
        elif ?? == 'px':
            mtx_elem = l*n*Vpp?? - l*n*Vpp??
        elif ?? == 'py':
            mtx_elem = m*n*Vpp?? - m*n*Vpp??
        elif ?? == 'pz':
            mtx_elem = n**2 * Vpp?? + (1.-n**2)*Vpp??
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3) * l*m*n*Vpd?? - 2*l*m*n*Vpd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3) * n**2 * m*Vpd?? + m*(1-2*n**2)*Vpd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3) * n**2 * l*Vpd?? + l*(1-2*n**2)*Vpd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*n*(l**2 - m**2)*Vpd?? - n*(l**2 - m**2)*Vpd??
        elif ?? == 'z2mr2':
            mtx_elem = n*(n**2 - (l**2 + m**2)/2.)*Vpd?? + np.sqrt(3)*n*(l**2 + m**2)*Vpd??
        elif ?? == 's2':
            mtx_elem = -n*Vps2??
        else:
            print("Error! pz")
            sys.exit()

    elif (?? == 'xy'):
        if ?? == 's':
            mtx_elem = np.sqrt(3)*l*m*Vds??
        elif ?? == 'px':
            mtx_elem = -1*(np.sqrt(3) * l**2 * m * Vdp?? + m*(1 - 2*l**2)*Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*(np.sqrt(3) * m**2 * l * Vdp?? + l*(1-2*m**2)*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*(np.sqrt(3) * l*m*n*Vdp?? - 2*l*m*n*Vdp??)
        elif ?? == 'xy':
            mtx_elem = 3 * l**2 * m**2 * Vdd?? + (l**2 + m**2 - 4 * l**2 * m**2) * Vdd?? + (n**2 + l**2 * m**2) * Vdd??
        elif ?? == 'yz':
            mtx_elem = 3 * l * m**2 * n * Vdd?? + l*n*(1-4*m**2)*Vdd?? + l*n*(m**2 - 1)*Vdd??
        elif ?? == 'zx':
            mtx_elem = 3 * l**2 * m * n * Vdd?? + m*n*(1-4*l**2)*Vdd?? + m*n*(l**2 - 1)*Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vdd?? + 2*l*m*(m**2 - l**2) * Vdd?? + l*m*(l**2 - m**2)/2. * Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vdd?? - 2*np.sqrt(3)*l*m * n**2 * Vdd?? + np.sqrt(3)*l*m*(1+n**2)/2. * Vdd??
        elif ?? == 's2':
            mtx_elem = np.sqrt(3)*l*m*Vds2??
        else:
            print("Error! xy")
            sys.exit()

    elif (?? == 'yz'):
        if ?? == 's':
            mtx_elem = np.sqrt(3)*m*n*Vds??
        elif ?? == 'px':
            mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdp?? - 2*l*m*n*Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*(np.sqrt(3) * m**2 * n * Vdp?? + n*(1 - 2*m**2)*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*(np.sqrt(3) * n**2 * m*Vdp?? + m*(1-2*n**2)*Vdp??)
        elif ?? == 'xy':
            mtx_elem = 3 * l * m**2 * n * Vdd?? + l*n*(1-4*m**2)*Vdd?? + l*n*(m**2 - 1)*Vdd??
        elif ?? == 'yz':
            mtx_elem = 3 * m**2 * n**2 * Vdd?? + (m**2 + n**2 - 4 * m**2 * n**2)*Vdd?? + (l**2 + m**2 * n**2)*Vdd??
        elif ?? == 'zx':
            mtx_elem = 3 * l * m * n**2 * Vdd?? + l*m*(1-4*n**2)*Vdd?? + l*m*(n**2 - 1)*Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vdd?? - m*n*(1 + 2*(l**2 - m**2))*Vdd?? + m*n*(1 + (l**2 - m**2)/2.)*Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vdd?? - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vdd??
        elif ?? == 's2':
            mtx_elem = np.sqrt(3)*m*n*Vds2??
        else:
            print("Error! yz")
            sys.exit()

    elif (?? == 'zx'):
        if ?? == 's':
            mtx_elem = np.sqrt(3)*l*n*Vds??
        elif ?? == 'px':
            mtx_elem = -1*(np.sqrt(3) * l**2 * n * Vdp?? + n*(1 - 2*l**2)*Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdp?? - 2*l*m*n*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*(np.sqrt(3) * n**2 * l*Vdp?? + l*(1-2*n**2)*Vdp??)
        elif ?? == 'xy':
            mtx_elem = 3 * l**2 * m * n * Vdd?? + m*n*(1-4*l**2)*Vdd?? + m*n*(l**2 - 1)*Vdd??
        elif ?? == 'yz':
            mtx_elem = 3 * l * m * n**2 * Vdd?? + l*m*(1-4*n**2)*Vdd?? + l*m*(n**2 - 1)*Vdd??
        elif ?? == 'zx':
            mtx_elem = 3 * l**2 * n**2 * Vdd?? + (n**2 + l**2 - 4 * l**2 * n**2)*Vdd?? + (m**2 + l**2 * n**2)*Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vdd?? + n*l*(1 - 2*(l**2 - m**2))*Vdd?? - n*l*(1 - (l**2 - m**2)/2.)*Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vdd?? - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vdd??
        elif ?? == 's2':
            mtx_elem = np.sqrt(3)*l*n*Vds2??
        else:
            print("Error! zx")
            sys.exit()

    elif (?? == 'x2my2'):
        if ?? == 's':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vds??
        elif ?? == 'px':
            mtx_elem = -1*((np.sqrt(3)/2.)*l*(l**2 - m**2) * Vdp?? + l*(1 - l**2 + m**2) * Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*((np.sqrt(3)/2.)*m*(l**2 - m**2)*Vdp?? - m*(1 +  l**2 - m**2)*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*((np.sqrt(3)/2.)*n*(l**2 - m**2)*Vdp?? - n*(l**2 - m**2)*Vdp??)
        elif ?? == 'xy':
            mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vdd?? + 2*l*m*(m**2 - l**2) * Vdd?? + l*m*(l**2 - m**2)/2. * Vdd??
        elif ?? == 'yz':
            mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vdd?? - m*n*(1 + 2*(l**2 - m**2))*Vdd?? + m*n*(1 + (l**2 - m**2)/2.)*Vdd??
        elif ?? == 'zx':
            mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vdd?? + n*l*(1 - 2*(l**2 - m**2))*Vdd?? - n*l*(1 - (l**2 - m**2)/2.)*Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (3./4.)*(l**2 - m**2)*Vdd?? + (l**2 + m**2 - (l**2 - m**2)**2)*Vdd?? + (n**2 + (l**2 - m**2)**2/4.)*Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3) * n**2 * (m**2 - l**2)*Vdd?? + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vdd??
        elif ?? == 's2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vds2??
        else:
            print("Error! x2my2")
            sys.exit()

    elif (?? == 'z2mr2'):
        if ?? == 's':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vds??
        elif ?? == 'px':
            mtx_elem = -1*(l*(n**2 - (l**2 + m**2)/2.)*Vdp?? - np.sqrt(3) * l * n**2 * Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*(m*(n**2 - (l**2 + m**2)/2.)*Vdp?? - np.sqrt(3)*m*n**2*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*(n*(n**2 - (l**2 + m**2)/2.)*Vdp?? + np.sqrt(3)*n*(l**2 + m**2)*Vdp??)
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vdd?? - 2*np.sqrt(3)*l*m * n**2 * Vdd?? + np.sqrt(3)*l*m*(1+n**2)/2. * Vdd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vdd?? - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vdd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vdd?? - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3) * n**2 * (m**2 - l**2)*Vdd?? + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)**2 * Vdd?? + 3 * n**2 * (l**2 + m**2)*Vdd?? + (3./4.)*(l**2 + m**2)**2 * Vdd??
        elif ?? == 's2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vds2??
        else:
            print("Error! z2mr2")
            sys.exit()

    elif (?? == 's2'):
        if ?? == 's':
            mtx_elem = Vs2s??
        elif ?? == 'px':
            mtx_elem = l*Vs2p??
        elif ?? == 'py':
            mtx_elem = m*Vs2p??
        elif ?? == 'pz':
            mtx_elem = n*Vs2p??
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3)*l*m*Vs2d??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3)*m*n*Vs2d??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3)*l*n*Vs2d??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2) * Vs2d??
        elif ?? == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vs2d??
        elif ?? == 's2':
            mtx_elem = Vs2s2??
        else:
            print("Error! s2")
            sys.exit()

    else:
        print("Error! ??")

    return mtx_elem

def calc_interatomic_mtx_element_Si_Si_Vogl(??,??,l,m,n,d):
    ### Calculates the interatomic matrix element between the
    ### orbital ?? on the reference Si atom and the ?? orbital on
    ### the target Si atom
    ###     * l,m,n are the directional cosines of the vector
    ###       going from the reference to the target atom
    ###     * d is the distance between atoms
    ###     * Using the sp3s* basis of Vogl (1981)
    ###         * Also excludes any strain effect

    d_Si = 5.431 * np.sqrt(3.)/4.
    d_ratio = (d_Si/d) # ratio of unstrained and strained inter-atomic distance

    ### Unstrained band parameters
    Es = -2.55247 # + E_offset
    Ep = 4.48593 # + E_offset
    Es2 = 23.44607 # + E_offset

    Vss?? = -1.86600
    Vsp?? = 2.91067
    Vss2?? = -1.39107

    Vpp?? = 4.08481
    Vpp?? = -1.49207

    Vs2p?? = 3.06822
    Vs2s2?? = -4.51331

    Delta_SO = 3*0.01851


    ### Calculate the renormalized matrix elements
    Vss?? = Vss?? * (d_ratio**nss??)
    Vss2?? = Vss2?? * (d_ratio**nss2??)
    Vs2s?? = Vss2?? * (d_ratio**nss2??)
    Vsp?? = Vsp?? * (d_ratio**nsp??)
    Vps?? = Vsp?? * (d_ratio**nsp??)
    Vsd?? = Vsd?? * (d_ratio**nsd??)
    Vds?? = Vsd?? * (d_ratio**nsd??)

    Vs2s2?? = Vs2s2?? * (d_ratio**ns2s2??)
    Vs2p?? = Vs2p?? * (d_ratio**ns2p??)
    Vps2?? = Vs2p?? * (d_ratio**ns2p??)
    Vs2d?? = Vs2d?? * (d_ratio**ns2d??)
    Vds2?? = Vs2d?? * (d_ratio**ns2d??)

    Vpp?? = Vpp?? * (d_ratio**npp??)
    Vpp?? = Vpp?? * (d_ratio**npp??)
    Vpd?? = Vpd?? * (d_ratio**npd??)
    Vdp?? = Vpd?? * (d_ratio**npd??)
    Vpd?? = Vpd?? * (d_ratio**npd??)
    Vdp?? = Vpd?? * (d_ratio**npd??)

    Vdd?? = Vdd?? * (d_ratio**ndd??)
    Vdd?? = Vdd?? * (d_ratio**ndd??)
    Vdd?? = Vdd?? * (d_ratio**ndd??)


    if (?? == 's'):
        if (?? == 's'):
            mtx_elem = Vss??
        elif (?? == 'px'):
            mtx_elem = l*Vsp??
        elif (?? == 'py'):
            mtx_elem = m*Vsp??
        elif (?? == 'pz'):
            mtx_elem = n*Vsp??
        elif (?? == 'xy'):
            mtx_elem = np.sqrt(3)*l*m*Vsd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3)*m*n*Vsd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3)*l*n*Vsd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vsd??
        elif ?? == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vsd??
        elif ?? == 's2':
            mtx_elem = Vss2??
        else:
            print("Error! s")
            sys.exit()

    elif (?? == 'px'):
        if ?? == 's':
            mtx_elem = -l*Vps??
        elif ?? == 'px':
            mtx_elem = l**2 * Vpp?? + (1 - l**2) * Vpp??
        elif ?? == 'py':
            mtx_elem = l*m*Vpp?? - l*m*Vpp??
        elif ?? == 'pz':
            mtx_elem = l*n*Vpp?? - l*n*Vpp??
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3) * l**2 * m * Vpd?? + m*(1 - 2*l**2)*Vpd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3) * l*m*n * Vpd?? - 2*l*m*n*Vpd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3) * l**2 * n * Vpd?? + n*(1 - 2*l**2)*Vpd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*l*(l**2 - m**2) * Vpd?? + l*(1 - l**2 + m**2) * Vpd??
        elif ?? == 'z2mr2':
            mtx_elem = l*(n**2 - (l**2 + m**2)/2.)*Vpd?? - np.sqrt(3) * l * n**2 * Vpd??
        elif ?? == 's2':
            mtx_elem = -l*Vps2??
        else:
            print("Error! px")
            sys.exit()

    elif (?? == 'py'):
        if ?? == 's':
            mtx_elem = -m*Vps??
        elif ?? == 'px':
            mtx_elem = l*m*Vpp?? - l*m*Vpp??
        elif ?? == 'py':
            mtx_elem = m**2 * Vpp?? + (1. - m**2)*Vpp??
        elif ?? == 'pz':
            mtx_elem = m*n*Vpp?? - m*n*Vpp??
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3) * m**2 * l * Vpd?? + l*(1-2*m**2)*Vpd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3) * m**2 * n * Vpd?? + n*(1 - 2*m**2)*Vpd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3) * l*m*n * Vpd?? - 2*l*m*n*Vpd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*m*(l**2 - m**2)*Vpd?? - m*(1 +  l**2 - m**2)*Vpd??
        elif ?? == 'z2mr2':
            mtx_elem = m*(n**2 - (l**2 + m**2)/2.)*Vpd?? - np.sqrt(3)*m*n**2*Vpd??
        elif ?? == 's2':
            mtx_elem = -m*Vps2??
        else:
            print("Error! py")
            sys.exit()

    elif (?? == 'pz'):
        if ?? == 's':
            mtx_elem = -n*Vps??
        elif ?? == 'px':
            mtx_elem = l*n*Vpp?? - l*n*Vpp??
        elif ?? == 'py':
            mtx_elem = m*n*Vpp?? - m*n*Vpp??
        elif ?? == 'pz':
            mtx_elem = n**2 * Vpp?? + (1.-n**2)*Vpp??
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3) * l*m*n*Vpd?? - 2*l*m*n*Vpd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3) * n**2 * m*Vpd?? + m*(1-2*n**2)*Vpd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3) * n**2 * l*Vpd?? + l*(1-2*n**2)*Vpd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*n*(l**2 - m**2)*Vpd?? - n*(l**2 - m**2)*Vpd??
        elif ?? == 'z2mr2':
            mtx_elem = n*(n**2 - (l**2 + m**2)/2.)*Vpd?? + np.sqrt(3)*n*(l**2 + m**2)*Vpd??
        elif ?? == 's2':
            mtx_elem = -n*Vps2??
        else:
            print("Error! pz")
            sys.exit()

    elif (?? == 'xy'):
        if ?? == 's':
            mtx_elem = np.sqrt(3)*l*m*Vds??
        elif ?? == 'px':
            mtx_elem = -1*(np.sqrt(3) * l**2 * m * Vdp?? + m*(1 - 2*l**2)*Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*(np.sqrt(3) * m**2 * l * Vdp?? + l*(1-2*m**2)*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*(np.sqrt(3) * l*m*n*Vdp?? - 2*l*m*n*Vdp??)
        elif ?? == 'xy':
            mtx_elem = 3 * l**2 * m**2 * Vdd?? + (l**2 + m**2 - 4 * l**2 * m**2) * Vdd?? + (n**2 + l**2 * m**2) * Vdd??
        elif ?? == 'yz':
            mtx_elem = 3 * l * m**2 * n * Vdd?? + l*n*(1-4*m**2)*Vdd?? + l*n*(m**2 - 1)*Vdd??
        elif ?? == 'zx':
            mtx_elem = 3 * l**2 * m * n * Vdd?? + m*n*(1-4*l**2)*Vdd?? + m*n*(l**2 - 1)*Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vdd?? + 2*l*m*(m**2 - l**2) * Vdd?? + l*m*(l**2 - m**2)/2. * Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vdd?? - 2*np.sqrt(3)*l*m * n**2 * Vdd?? + np.sqrt(3)*l*m*(1+n**2)/2. * Vdd??
        elif ?? == 's2':
            mtx_elem = np.sqrt(3)*l*m*Vds2??
        else:
            print("Error! xy")
            sys.exit()

    elif (?? == 'yz'):
        if ?? == 's':
            mtx_elem = np.sqrt(3)*m*n*Vds??
        elif ?? == 'px':
            mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdp?? - 2*l*m*n*Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*(np.sqrt(3) * m**2 * n * Vdp?? + n*(1 - 2*m**2)*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*(np.sqrt(3) * n**2 * m*Vdp?? + m*(1-2*n**2)*Vdp??)
        elif ?? == 'xy':
            mtx_elem = 3 * l * m**2 * n * Vdd?? + l*n*(1-4*m**2)*Vdd?? + l*n*(m**2 - 1)*Vdd??
        elif ?? == 'yz':
            mtx_elem = 3 * m**2 * n**2 * Vdd?? + (m**2 + n**2 - 4 * m**2 * n**2)*Vdd?? + (l**2 + m**2 * n**2)*Vdd??
        elif ?? == 'zx':
            mtx_elem = 3 * l * m * n**2 * Vdd?? + l*m*(1-4*n**2)*Vdd?? + l*m*(n**2 - 1)*Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vdd?? - m*n*(1 + 2*(l**2 - m**2))*Vdd?? + m*n*(1 + (l**2 - m**2)/2.)*Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vdd?? - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vdd??
        elif ?? == 's2':
            mtx_elem = np.sqrt(3)*m*n*Vds2??
        else:
            print("Error! yz")
            sys.exit()

    elif (?? == 'zx'):
        if ?? == 's':
            mtx_elem = np.sqrt(3)*l*n*Vds??
        elif ?? == 'px':
            mtx_elem = -1*(np.sqrt(3) * l**2 * n * Vdp?? + n*(1 - 2*l**2)*Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdp?? - 2*l*m*n*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*(np.sqrt(3) * n**2 * l*Vdp?? + l*(1-2*n**2)*Vdp??)
        elif ?? == 'xy':
            mtx_elem = 3 * l**2 * m * n * Vdd?? + m*n*(1-4*l**2)*Vdd?? + m*n*(l**2 - 1)*Vdd??
        elif ?? == 'yz':
            mtx_elem = 3 * l * m * n**2 * Vdd?? + l*m*(1-4*n**2)*Vdd?? + l*m*(n**2 - 1)*Vdd??
        elif ?? == 'zx':
            mtx_elem = 3 * l**2 * n**2 * Vdd?? + (n**2 + l**2 - 4 * l**2 * n**2)*Vdd?? + (m**2 + l**2 * n**2)*Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vdd?? + n*l*(1 - 2*(l**2 - m**2))*Vdd?? - n*l*(1 - (l**2 - m**2)/2.)*Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vdd?? - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vdd??
        elif ?? == 's2':
            mtx_elem = np.sqrt(3)*l*n*Vds2??
        else:
            print("Error! zx")
            sys.exit()

    elif (?? == 'x2my2'):
        if ?? == 's':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vds??
        elif ?? == 'px':
            mtx_elem = -1*((np.sqrt(3)/2.)*l*(l**2 - m**2) * Vdp?? + l*(1 - l**2 + m**2) * Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*((np.sqrt(3)/2.)*m*(l**2 - m**2)*Vdp?? - m*(1 +  l**2 - m**2)*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*((np.sqrt(3)/2.)*n*(l**2 - m**2)*Vdp?? - n*(l**2 - m**2)*Vdp??)
        elif ?? == 'xy':
            mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vdd?? + 2*l*m*(m**2 - l**2) * Vdd?? + l*m*(l**2 - m**2)/2. * Vdd??
        elif ?? == 'yz':
            mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vdd?? - m*n*(1 + 2*(l**2 - m**2))*Vdd?? + m*n*(1 + (l**2 - m**2)/2.)*Vdd??
        elif ?? == 'zx':
            mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vdd?? + n*l*(1 - 2*(l**2 - m**2))*Vdd?? - n*l*(1 - (l**2 - m**2)/2.)*Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (3./4.)*(l**2 - m**2)*Vdd?? + (l**2 + m**2 - (l**2 - m**2)**2)*Vdd?? + (n**2 + (l**2 - m**2)**2/4.)*Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3) * n**2 * (m**2 - l**2)*Vdd?? + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vdd??
        elif ?? == 's2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vds2??
        else:
            print("Error! x2my2")
            sys.exit()

    elif (?? == 'z2mr2'):
        if ?? == 's':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vds??
        elif ?? == 'px':
            mtx_elem = -1*(l*(n**2 - (l**2 + m**2)/2.)*Vdp?? - np.sqrt(3) * l * n**2 * Vdp??)
        elif ?? == 'py':
            mtx_elem = -1*(m*(n**2 - (l**2 + m**2)/2.)*Vdp?? - np.sqrt(3)*m*n**2*Vdp??)
        elif ?? == 'pz':
            mtx_elem = -1*(n*(n**2 - (l**2 + m**2)/2.)*Vdp?? + np.sqrt(3)*n*(l**2 + m**2)*Vdp??)
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vdd?? - 2*np.sqrt(3)*l*m * n**2 * Vdd?? + np.sqrt(3)*l*m*(1+n**2)/2. * Vdd??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vdd?? - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vdd??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vdd?? - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vdd??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vdd?? + np.sqrt(3) * n**2 * (m**2 - l**2)*Vdd?? + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vdd??
        elif ?? == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)**2 * Vdd?? + 3 * n**2 * (l**2 + m**2)*Vdd?? + (3./4.)*(l**2 + m**2)**2 * Vdd??
        elif ?? == 's2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vds2??
        else:
            print("Error! z2mr2")
            sys.exit()

    elif (?? == 's2'):
        if ?? == 's':
            mtx_elem = Vs2s??
        elif ?? == 'px':
            mtx_elem = l*Vs2p??
        elif ?? == 'py':
            mtx_elem = m*Vs2p??
        elif ?? == 'pz':
            mtx_elem = n*Vs2p??
        elif ?? == 'xy':
            mtx_elem = np.sqrt(3)*l*m*Vs2d??
        elif ?? == 'yz':
            mtx_elem = np.sqrt(3)*m*n*Vs2d??
        elif ?? == 'zx':
            mtx_elem = np.sqrt(3)*l*n*Vs2d??
        elif ?? == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2) * Vs2d??
        elif ?? == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vs2d??
        elif ?? == 's2':
            mtx_elem = Vs2s2??
        else:
            print("Error! s2")
            sys.exit()

    else:
        print("Error! ??")

    return mtx_elem
