"""
    Sub object to handle the generation of Germanium concentration profiles
    for the parent instance of SiGe_Quantum_Dot
"""

import numpy as np
import parameters as par
import matplotlib.pyplot as plt

class GeProfile_subObject:

    def __init__(self,QD_obj):
        self.QD_obj = QD_obj # parent instance of SiGe_Quantum_Dot

    def uniform_profile(self,N_bar,N_well,xGe_bar,xGe_well,PLOT = False):
        ### Generates a Ge_conc_arr with N_bar atomic layers in the barriers
        ### and N_well atomic layers in the well region.
        ###     # The concentrations are constant in the two regions


        Ge_conc_arr = np.zeros(2*N_bar+N_well)
        Ge_conc_arr[:N_bar] = xGe_bar
        Ge_conc_arr[-N_bar:] = xGe_bar
        Ge_conc_arr[N_bar:N_bar+N_well] = xGe_well

        if PLOT == True:
            self.plot_profile(Ge_conc_arr)

        return Ge_conc_arr

    def uniform_profile_gradedInterface(self,N_bar,N_well,N_intface,xGe_bar,xGe_well,PLOT = False):
        ### Generates a Ge_arr_Full with N_bar atomic layers in the barriers
        ### and N_well atomic layers in the well region.
        ###     * The concentrations are constant in the two regions
        ###     * There are interface regions between the constant Ge regions where the Ge concentration
        ###       varies linearly

        Ge_conc_arr = np.zeros(2*N_bar+N_well + 2*N_intface)
        Ge_conc_arr[:N_bar] = xGe_bar
        Ge_conc_arr[-N_bar:] = xGe_bar
        Ge_conc_arr[N_bar+N_intface:N_bar+N_intface+N_well] = xGe_well

        step = (xGe_bar - xGe_well)/(N_intface+1.)
        for i in range(N_intface):
            Ge_conc_arr[N_bar+i] = xGe_bar - (i+1)*step
            Ge_conc_arr[N_bar+N_intface+N_well+i] = xGe_well + (i+1)*step

        if PLOT == True:
            self.plot_profile(Ge_conc_arr)

        return Ge_conc_arr    


    def plot_profile(self,Ge_conc_arr,z_arr = -1):
        ### Plot the Germanium profile with nice labels and such

        if (type(z_arr) == int):
            X = np.arange(1,Ge_conc_arr.size +1)
            x_label = "layer index"
        else:
            X = z_arr[:]/10.
            x_label = r"$z$ (nm)"

        fig = plt.figure()
        width = 3.847; height = .5 * width
        fig.set_size_inches(width,height)
        ax1 = fig.add_subplot(1,1,1)
        ax1.scatter(X,Ge_conc_arr, c = 'k',s = 5)
        #ax1.scatter(X,Ge_conc_arr[::-1], c = 'r',s = 3)
        ax1.set_xlabel(x_label,fontsize = 12)
        ax1.set_ylabel(r"$n_{Ge}$",fontsize = 12)
        ax1.set_xlim(xmin = min(1.0*np.min(X),0),xmax = 1.0*np.max(X))
        ax1.grid()
        ax1.set_ylim(ymin = min(-.05*np.max(Ge_conc_arr),-0.01))
        plt.subplots_adjust(left = 0.16, bottom=0.28, right=0.95, top=0.96)
        plt.show()
