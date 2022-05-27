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






def calc_interatomic_mtx_element_Si_Si(α,β,l,m,n,d):
    ### Calculates the interatomic matrix element between the
    ### orbital β on the reference Si atom and the α orbital on
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

    Vssσ = -1.86600
    Vspσ = 2.91067
    Vsdσ = -2.23992
    Vss2σ = -1.39107

    Vppσ = 4.08481
    Vppπ = -1.49207
    Vpdσ = -1.66657
    Vpdπ = 2.39936

    Vddσ = -1.82945
    Vddπ = 3.08177
    Vddδ = -1.56676

    Vs2pσ = 3.06822
    Vs2dσ = -0.77711
    Vs2s2σ = -4.51331

    Delta_SO = 3*0.01851

    ### Deformation exponentials
    nssσ = 3.56701
    nss2σ = 1.51967
    nspσ = 2.03530
    nsdσ = 2.14811

    ns2s2σ = 0.64401
    ns2pσ = 1.46652
    ns2dσ = 1.79667

    nppσ = 2.01907
    nppπ = 2.87276
    npdσ = 1.00446
    npdπ = 1.78029

    nddσ = 1.73865
    nddπ = 1.80442
    nddδ = 2.54691
    #bd = 0.443

    ### Calculate the renormalized matrix elements
    Vssσ = Vssσ * (d_ratio**nssσ)
    Vss2σ = Vss2σ * (d_ratio**nss2σ)
    Vs2sσ = Vss2σ * (d_ratio**nss2σ)
    Vspσ = Vspσ * (d_ratio**nspσ)
    Vpsσ = Vspσ * (d_ratio**nspσ)
    Vsdσ = Vsdσ * (d_ratio**nsdσ)
    Vdsσ = Vsdσ * (d_ratio**nsdσ)

    Vs2s2σ = Vs2s2σ * (d_ratio**ns2s2σ)
    Vs2pσ = Vs2pσ * (d_ratio**ns2pσ)
    Vps2σ = Vs2pσ * (d_ratio**ns2pσ)
    Vs2dσ = Vs2dσ * (d_ratio**ns2dσ)
    Vds2σ = Vs2dσ * (d_ratio**ns2dσ)

    Vppσ = Vppσ * (d_ratio**nppσ)
    Vppπ = Vppπ * (d_ratio**nppπ)
    Vpdσ = Vpdσ * (d_ratio**npdσ)
    Vdpσ = Vpdσ * (d_ratio**npdσ)
    Vpdπ = Vpdπ * (d_ratio**npdπ)
    Vdpπ = Vpdπ * (d_ratio**npdπ)

    Vddσ = Vddσ * (d_ratio**nddσ)
    Vddπ = Vddπ * (d_ratio**nddπ)
    Vddδ = Vddδ * (d_ratio**nddδ)


    if (α == 's'):
        if (β == 's'):
            mtx_elem = Vssσ
        elif (β == 'px'):
            mtx_elem = l*Vspσ
        elif (β == 'py'):
            mtx_elem = m*Vspσ
        elif (β == 'pz'):
            mtx_elem = n*Vspσ
        elif (β == 'xy'):
            mtx_elem = np.sqrt(3)*l*m*Vsdσ
        elif β == 'yz':
            mtx_elem = np.sqrt(3)*m*n*Vsdσ
        elif β == 'zx':
            mtx_elem = np.sqrt(3)*l*n*Vsdσ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vsdσ
        elif β == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vsdσ
        elif β == 's2':
            mtx_elem = Vss2σ
        else:
            print("Error! s")
            sys.exit()

    elif (α == 'px'):
        if β == 's':
            mtx_elem = -l*Vpsσ
        elif β == 'px':
            mtx_elem = l**2 * Vppσ + (1 - l**2) * Vppπ
        elif β == 'py':
            mtx_elem = l*m*Vppσ - l*m*Vppπ
        elif β == 'pz':
            mtx_elem = l*n*Vppσ - l*n*Vppπ
        elif β == 'xy':
            mtx_elem = np.sqrt(3) * l**2 * m * Vpdσ + m*(1 - 2*l**2)*Vpdπ
        elif β == 'yz':
            mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
        elif β == 'zx':
            mtx_elem = np.sqrt(3) * l**2 * n * Vpdσ + n*(1 - 2*l**2)*Vpdπ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*l*(l**2 - m**2) * Vpdσ + l*(1 - l**2 + m**2) * Vpdπ
        elif β == 'z2mr2':
            mtx_elem = l*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3) * l * n**2 * Vpdπ
        elif β == 's2':
            mtx_elem = -l*Vps2σ
        else:
            print("Error! px")
            sys.exit()

    elif (α == 'py'):
        if β == 's':
            mtx_elem = -m*Vpsσ
        elif β == 'px':
            mtx_elem = l*m*Vppσ - l*m*Vppπ
        elif β == 'py':
            mtx_elem = m**2 * Vppσ + (1. - m**2)*Vppπ
        elif β == 'pz':
            mtx_elem = m*n*Vppσ - m*n*Vppπ
        elif β == 'xy':
            mtx_elem = np.sqrt(3) * m**2 * l * Vpdσ + l*(1-2*m**2)*Vpdπ
        elif β == 'yz':
            mtx_elem = np.sqrt(3) * m**2 * n * Vpdσ + n*(1 - 2*m**2)*Vpdπ
        elif β == 'zx':
            mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*m*(l**2 - m**2)*Vpdσ - m*(1 +  l**2 - m**2)*Vpdπ
        elif β == 'z2mr2':
            mtx_elem = m*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3)*m*n**2*Vpdπ
        elif β == 's2':
            mtx_elem = -m*Vps2σ
        else:
            print("Error! py")
            sys.exit()

    elif (α == 'pz'):
        if β == 's':
            mtx_elem = -n*Vpsσ
        elif β == 'px':
            mtx_elem = l*n*Vppσ - l*n*Vppπ
        elif β == 'py':
            mtx_elem = m*n*Vppσ - m*n*Vppπ
        elif β == 'pz':
            mtx_elem = n**2 * Vppσ + (1.-n**2)*Vppπ
        elif β == 'xy':
            mtx_elem = np.sqrt(3) * l*m*n*Vpdσ - 2*l*m*n*Vpdπ
        elif β == 'yz':
            mtx_elem = np.sqrt(3) * n**2 * m*Vpdσ + m*(1-2*n**2)*Vpdπ
        elif β == 'zx':
            mtx_elem = np.sqrt(3) * n**2 * l*Vpdσ + l*(1-2*n**2)*Vpdπ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*n*(l**2 - m**2)*Vpdσ - n*(l**2 - m**2)*Vpdπ
        elif β == 'z2mr2':
            mtx_elem = n*(n**2 - (l**2 + m**2)/2.)*Vpdσ + np.sqrt(3)*n*(l**2 + m**2)*Vpdπ
        elif β == 's2':
            mtx_elem = -n*Vps2σ
        else:
            print("Error! pz")
            sys.exit()

    elif (α == 'xy'):
        if β == 's':
            mtx_elem = np.sqrt(3)*l*m*Vdsσ
        elif β == 'px':
            mtx_elem = -1*(np.sqrt(3) * l**2 * m * Vdpσ + m*(1 - 2*l**2)*Vdpπ)
        elif β == 'py':
            mtx_elem = -1*(np.sqrt(3) * m**2 * l * Vdpσ + l*(1-2*m**2)*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*(np.sqrt(3) * l*m*n*Vdpσ - 2*l*m*n*Vdpπ)
        elif β == 'xy':
            mtx_elem = 3 * l**2 * m**2 * Vddσ + (l**2 + m**2 - 4 * l**2 * m**2) * Vddπ + (n**2 + l**2 * m**2) * Vddδ
        elif β == 'yz':
            mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
        elif β == 'zx':
            mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
        elif β == 'x2my2':
            mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
        elif β == 'z2mr2':
            mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
        elif β == 's2':
            mtx_elem = np.sqrt(3)*l*m*Vds2σ
        else:
            print("Error! xy")
            sys.exit()

    elif (α == 'yz'):
        if β == 's':
            mtx_elem = np.sqrt(3)*m*n*Vdsσ
        elif β == 'px':
            mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
        elif β == 'py':
            mtx_elem = -1*(np.sqrt(3) * m**2 * n * Vdpσ + n*(1 - 2*m**2)*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*(np.sqrt(3) * n**2 * m*Vdpσ + m*(1-2*n**2)*Vdpπ)
        elif β == 'xy':
            mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
        elif β == 'yz':
            mtx_elem = 3 * m**2 * n**2 * Vddσ + (m**2 + n**2 - 4 * m**2 * n**2)*Vddπ + (l**2 + m**2 * n**2)*Vddδ
        elif β == 'zx':
            mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
        elif β == 'x2my2':
            mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
        elif β == 'z2mr2':
            mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
        elif β == 's2':
            mtx_elem = np.sqrt(3)*m*n*Vds2σ
        else:
            print("Error! yz")
            sys.exit()

    elif (α == 'zx'):
        if β == 's':
            mtx_elem = np.sqrt(3)*l*n*Vdsσ
        elif β == 'px':
            mtx_elem = -1*(np.sqrt(3) * l**2 * n * Vdpσ + n*(1 - 2*l**2)*Vdpπ)
        elif β == 'py':
            mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*(np.sqrt(3) * n**2 * l*Vdpσ + l*(1-2*n**2)*Vdpπ)
        elif β == 'xy':
            mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
        elif β == 'yz':
            mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
        elif β == 'zx':
            mtx_elem = 3 * l**2 * n**2 * Vddσ + (n**2 + l**2 - 4 * l**2 * n**2)*Vddπ + (m**2 + l**2 * n**2)*Vddδ
        elif β == 'x2my2':
            mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
        elif β == 'z2mr2':
            mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
        elif β == 's2':
            mtx_elem = np.sqrt(3)*l*n*Vds2σ
        else:
            print("Error! zx")
            sys.exit()

    elif (α == 'x2my2'):
        if β == 's':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vdsσ
        elif β == 'px':
            mtx_elem = -1*((np.sqrt(3)/2.)*l*(l**2 - m**2) * Vdpσ + l*(1 - l**2 + m**2) * Vdpπ)
        elif β == 'py':
            mtx_elem = -1*((np.sqrt(3)/2.)*m*(l**2 - m**2)*Vdpσ - m*(1 +  l**2 - m**2)*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*((np.sqrt(3)/2.)*n*(l**2 - m**2)*Vdpσ - n*(l**2 - m**2)*Vdpπ)
        elif β == 'xy':
            mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
        elif β == 'yz':
            mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
        elif β == 'zx':
            mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
        elif β == 'x2my2':
            mtx_elem = (3./4.)*(l**2 - m**2)*Vddσ + (l**2 + m**2 - (l**2 - m**2)**2)*Vddπ + (n**2 + (l**2 - m**2)**2/4.)*Vddδ
        elif β == 'z2mr2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
        elif β == 's2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vds2σ
        else:
            print("Error! x2my2")
            sys.exit()

    elif (α == 'z2mr2'):
        if β == 's':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vdsσ
        elif β == 'px':
            mtx_elem = -1*(l*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3) * l * n**2 * Vdpπ)
        elif β == 'py':
            mtx_elem = -1*(m*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3)*m*n**2*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*(n*(n**2 - (l**2 + m**2)/2.)*Vdpσ + np.sqrt(3)*n*(l**2 + m**2)*Vdpπ)
        elif β == 'xy':
            mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
        elif β == 'yz':
            mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
        elif β == 'zx':
            mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
        elif β == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)**2 * Vddσ + 3 * n**2 * (l**2 + m**2)*Vddπ + (3./4.)*(l**2 + m**2)**2 * Vddδ
        elif β == 's2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vds2σ
        else:
            print("Error! z2mr2")
            sys.exit()

    elif (α == 's2'):
        if β == 's':
            mtx_elem = Vs2sσ
        elif β == 'px':
            mtx_elem = l*Vs2pσ
        elif β == 'py':
            mtx_elem = m*Vs2pσ
        elif β == 'pz':
            mtx_elem = n*Vs2pσ
        elif β == 'xy':
            mtx_elem = np.sqrt(3)*l*m*Vs2dσ
        elif β == 'yz':
            mtx_elem = np.sqrt(3)*m*n*Vs2dσ
        elif β == 'zx':
            mtx_elem = np.sqrt(3)*l*n*Vs2dσ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2) * Vs2dσ
        elif β == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vs2dσ
        elif β == 's2':
            mtx_elem = Vs2s2σ
        else:
            print("Error! s2")
            sys.exit()

    else:
        print("Error! α")

    return mtx_elem


def calc_interatomic_mtx_element_Si_Si_Vogl(α,β,l,m,n,d):
    ### Calculates the interatomic matrix element between the
    ### orbital β on the reference Si atom and the α orbital on
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

    Vssσ = -1.86600
    Vspσ = 2.91067
    Vss2σ = -1.39107

    Vppσ = 4.08481
    Vppπ = -1.49207

    Vs2pσ = 3.06822
    Vs2s2σ = -4.51331

    Delta_SO = 3*0.01851


    ### Calculate the renormalized matrix elements
    Vssσ = Vssσ * (d_ratio**nssσ)
    Vss2σ = Vss2σ * (d_ratio**nss2σ)
    Vs2sσ = Vss2σ * (d_ratio**nss2σ)
    Vspσ = Vspσ * (d_ratio**nspσ)
    Vpsσ = Vspσ * (d_ratio**nspσ)
    Vsdσ = Vsdσ * (d_ratio**nsdσ)
    Vdsσ = Vsdσ * (d_ratio**nsdσ)

    Vs2s2σ = Vs2s2σ * (d_ratio**ns2s2σ)
    Vs2pσ = Vs2pσ * (d_ratio**ns2pσ)
    Vps2σ = Vs2pσ * (d_ratio**ns2pσ)
    Vs2dσ = Vs2dσ * (d_ratio**ns2dσ)
    Vds2σ = Vs2dσ * (d_ratio**ns2dσ)

    Vppσ = Vppσ * (d_ratio**nppσ)
    Vppπ = Vppπ * (d_ratio**nppπ)
    Vpdσ = Vpdσ * (d_ratio**npdσ)
    Vdpσ = Vpdσ * (d_ratio**npdσ)
    Vpdπ = Vpdπ * (d_ratio**npdπ)
    Vdpπ = Vpdπ * (d_ratio**npdπ)

    Vddσ = Vddσ * (d_ratio**nddσ)
    Vddπ = Vddπ * (d_ratio**nddπ)
    Vddδ = Vddδ * (d_ratio**nddδ)


    if (α == 's'):
        if (β == 's'):
            mtx_elem = Vssσ
        elif (β == 'px'):
            mtx_elem = l*Vspσ
        elif (β == 'py'):
            mtx_elem = m*Vspσ
        elif (β == 'pz'):
            mtx_elem = n*Vspσ
        elif (β == 'xy'):
            mtx_elem = np.sqrt(3)*l*m*Vsdσ
        elif β == 'yz':
            mtx_elem = np.sqrt(3)*m*n*Vsdσ
        elif β == 'zx':
            mtx_elem = np.sqrt(3)*l*n*Vsdσ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vsdσ
        elif β == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vsdσ
        elif β == 's2':
            mtx_elem = Vss2σ
        else:
            print("Error! s")
            sys.exit()

    elif (α == 'px'):
        if β == 's':
            mtx_elem = -l*Vpsσ
        elif β == 'px':
            mtx_elem = l**2 * Vppσ + (1 - l**2) * Vppπ
        elif β == 'py':
            mtx_elem = l*m*Vppσ - l*m*Vppπ
        elif β == 'pz':
            mtx_elem = l*n*Vppσ - l*n*Vppπ
        elif β == 'xy':
            mtx_elem = np.sqrt(3) * l**2 * m * Vpdσ + m*(1 - 2*l**2)*Vpdπ
        elif β == 'yz':
            mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
        elif β == 'zx':
            mtx_elem = np.sqrt(3) * l**2 * n * Vpdσ + n*(1 - 2*l**2)*Vpdπ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*l*(l**2 - m**2) * Vpdσ + l*(1 - l**2 + m**2) * Vpdπ
        elif β == 'z2mr2':
            mtx_elem = l*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3) * l * n**2 * Vpdπ
        elif β == 's2':
            mtx_elem = -l*Vps2σ
        else:
            print("Error! px")
            sys.exit()

    elif (α == 'py'):
        if β == 's':
            mtx_elem = -m*Vpsσ
        elif β == 'px':
            mtx_elem = l*m*Vppσ - l*m*Vppπ
        elif β == 'py':
            mtx_elem = m**2 * Vppσ + (1. - m**2)*Vppπ
        elif β == 'pz':
            mtx_elem = m*n*Vppσ - m*n*Vppπ
        elif β == 'xy':
            mtx_elem = np.sqrt(3) * m**2 * l * Vpdσ + l*(1-2*m**2)*Vpdπ
        elif β == 'yz':
            mtx_elem = np.sqrt(3) * m**2 * n * Vpdσ + n*(1 - 2*m**2)*Vpdπ
        elif β == 'zx':
            mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*m*(l**2 - m**2)*Vpdσ - m*(1 +  l**2 - m**2)*Vpdπ
        elif β == 'z2mr2':
            mtx_elem = m*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3)*m*n**2*Vpdπ
        elif β == 's2':
            mtx_elem = -m*Vps2σ
        else:
            print("Error! py")
            sys.exit()

    elif (α == 'pz'):
        if β == 's':
            mtx_elem = -n*Vpsσ
        elif β == 'px':
            mtx_elem = l*n*Vppσ - l*n*Vppπ
        elif β == 'py':
            mtx_elem = m*n*Vppσ - m*n*Vppπ
        elif β == 'pz':
            mtx_elem = n**2 * Vppσ + (1.-n**2)*Vppπ
        elif β == 'xy':
            mtx_elem = np.sqrt(3) * l*m*n*Vpdσ - 2*l*m*n*Vpdπ
        elif β == 'yz':
            mtx_elem = np.sqrt(3) * n**2 * m*Vpdσ + m*(1-2*n**2)*Vpdπ
        elif β == 'zx':
            mtx_elem = np.sqrt(3) * n**2 * l*Vpdσ + l*(1-2*n**2)*Vpdπ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*n*(l**2 - m**2)*Vpdσ - n*(l**2 - m**2)*Vpdπ
        elif β == 'z2mr2':
            mtx_elem = n*(n**2 - (l**2 + m**2)/2.)*Vpdσ + np.sqrt(3)*n*(l**2 + m**2)*Vpdπ
        elif β == 's2':
            mtx_elem = -n*Vps2σ
        else:
            print("Error! pz")
            sys.exit()

    elif (α == 'xy'):
        if β == 's':
            mtx_elem = np.sqrt(3)*l*m*Vdsσ
        elif β == 'px':
            mtx_elem = -1*(np.sqrt(3) * l**2 * m * Vdpσ + m*(1 - 2*l**2)*Vdpπ)
        elif β == 'py':
            mtx_elem = -1*(np.sqrt(3) * m**2 * l * Vdpσ + l*(1-2*m**2)*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*(np.sqrt(3) * l*m*n*Vdpσ - 2*l*m*n*Vdpπ)
        elif β == 'xy':
            mtx_elem = 3 * l**2 * m**2 * Vddσ + (l**2 + m**2 - 4 * l**2 * m**2) * Vddπ + (n**2 + l**2 * m**2) * Vddδ
        elif β == 'yz':
            mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
        elif β == 'zx':
            mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
        elif β == 'x2my2':
            mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
        elif β == 'z2mr2':
            mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
        elif β == 's2':
            mtx_elem = np.sqrt(3)*l*m*Vds2σ
        else:
            print("Error! xy")
            sys.exit()

    elif (α == 'yz'):
        if β == 's':
            mtx_elem = np.sqrt(3)*m*n*Vdsσ
        elif β == 'px':
            mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
        elif β == 'py':
            mtx_elem = -1*(np.sqrt(3) * m**2 * n * Vdpσ + n*(1 - 2*m**2)*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*(np.sqrt(3) * n**2 * m*Vdpσ + m*(1-2*n**2)*Vdpπ)
        elif β == 'xy':
            mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
        elif β == 'yz':
            mtx_elem = 3 * m**2 * n**2 * Vddσ + (m**2 + n**2 - 4 * m**2 * n**2)*Vddπ + (l**2 + m**2 * n**2)*Vddδ
        elif β == 'zx':
            mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
        elif β == 'x2my2':
            mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
        elif β == 'z2mr2':
            mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
        elif β == 's2':
            mtx_elem = np.sqrt(3)*m*n*Vds2σ
        else:
            print("Error! yz")
            sys.exit()

    elif (α == 'zx'):
        if β == 's':
            mtx_elem = np.sqrt(3)*l*n*Vdsσ
        elif β == 'px':
            mtx_elem = -1*(np.sqrt(3) * l**2 * n * Vdpσ + n*(1 - 2*l**2)*Vdpπ)
        elif β == 'py':
            mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*(np.sqrt(3) * n**2 * l*Vdpσ + l*(1-2*n**2)*Vdpπ)
        elif β == 'xy':
            mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
        elif β == 'yz':
            mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
        elif β == 'zx':
            mtx_elem = 3 * l**2 * n**2 * Vddσ + (n**2 + l**2 - 4 * l**2 * n**2)*Vddπ + (m**2 + l**2 * n**2)*Vddδ
        elif β == 'x2my2':
            mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
        elif β == 'z2mr2':
            mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
        elif β == 's2':
            mtx_elem = np.sqrt(3)*l*n*Vds2σ
        else:
            print("Error! zx")
            sys.exit()

    elif (α == 'x2my2'):
        if β == 's':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vdsσ
        elif β == 'px':
            mtx_elem = -1*((np.sqrt(3)/2.)*l*(l**2 - m**2) * Vdpσ + l*(1 - l**2 + m**2) * Vdpπ)
        elif β == 'py':
            mtx_elem = -1*((np.sqrt(3)/2.)*m*(l**2 - m**2)*Vdpσ - m*(1 +  l**2 - m**2)*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*((np.sqrt(3)/2.)*n*(l**2 - m**2)*Vdpσ - n*(l**2 - m**2)*Vdpπ)
        elif β == 'xy':
            mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
        elif β == 'yz':
            mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
        elif β == 'zx':
            mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
        elif β == 'x2my2':
            mtx_elem = (3./4.)*(l**2 - m**2)*Vddσ + (l**2 + m**2 - (l**2 - m**2)**2)*Vddπ + (n**2 + (l**2 - m**2)**2/4.)*Vddδ
        elif β == 'z2mr2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
        elif β == 's2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vds2σ
        else:
            print("Error! x2my2")
            sys.exit()

    elif (α == 'z2mr2'):
        if β == 's':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vdsσ
        elif β == 'px':
            mtx_elem = -1*(l*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3) * l * n**2 * Vdpπ)
        elif β == 'py':
            mtx_elem = -1*(m*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3)*m*n**2*Vdpπ)
        elif β == 'pz':
            mtx_elem = -1*(n*(n**2 - (l**2 + m**2)/2.)*Vdpσ + np.sqrt(3)*n*(l**2 + m**2)*Vdpπ)
        elif β == 'xy':
            mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
        elif β == 'yz':
            mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
        elif β == 'zx':
            mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
        elif β == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)**2 * Vddσ + 3 * n**2 * (l**2 + m**2)*Vddπ + (3./4.)*(l**2 + m**2)**2 * Vddδ
        elif β == 's2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vds2σ
        else:
            print("Error! z2mr2")
            sys.exit()

    elif (α == 's2'):
        if β == 's':
            mtx_elem = Vs2sσ
        elif β == 'px':
            mtx_elem = l*Vs2pσ
        elif β == 'py':
            mtx_elem = m*Vs2pσ
        elif β == 'pz':
            mtx_elem = n*Vs2pσ
        elif β == 'xy':
            mtx_elem = np.sqrt(3)*l*m*Vs2dσ
        elif β == 'yz':
            mtx_elem = np.sqrt(3)*m*n*Vs2dσ
        elif β == 'zx':
            mtx_elem = np.sqrt(3)*l*n*Vs2dσ
        elif β == 'x2my2':
            mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2) * Vs2dσ
        elif β == 'z2mr2':
            mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vs2dσ
        elif β == 's2':
            mtx_elem = Vs2s2σ
        else:
            print("Error! s2")
            sys.exit()

    else:
        print("Error! α")

    return mtx_elem
