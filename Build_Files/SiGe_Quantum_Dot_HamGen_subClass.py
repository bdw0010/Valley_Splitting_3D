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


    def 
