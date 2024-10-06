# ===========
# PARITY PLOT
# ===========

"""
Author: S. Falletta

This script produces parity plots corresponding to two set extxyz files, namely
the DFT input data and the ML evaluate data. Considered quantities are energy,
atomic positions, forces, polarization, Born charges, and polarizability.
MAE and RMSE are given for each quantity in each parity plot.

How to run:
• SiO2:   python3 parity_plot.py SiO2
• BaTiO3: python3 parity_plot.py BaTiO3

Before using this script, make sure your folders are organized as follows:
Nomenclature for input folder: {system} contains {system}.xyz and {system}-ML.xyz ,
where {system} is either SiO2 or BaTiO3.
"""

import sys
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24,'legend.fontsize': 20,'legend.handlelength': 0.5})
from matplotlib.ticker import (AutoMinorLocator)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error

def axis_settings(ax):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_tick_params(which='major', width=3.0, length=12, direction="in")
    ax.xaxis.set_tick_params(which='minor', width=3.0, length=6,  direction="in")
    ax.yaxis.set_tick_params(which='major', width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which='minor', width=3.0, length=6,  direction="in")
    ax.yaxis.set_ticks_position('both')

class extxyz:
    
    def split(self,string,chr):
        return ' '.join(string.split(chr))

    def __init__(self, filename):

        print("Parsing file {:s}".format(filename))

        with open(filename,"r") as f:
            data = f.readlines()
        self.nat = int(data[0].split()[0])
        nframes = int(len(data)/(self.nat+2))
        self.E = np.zeros(nframes)
        self.R = np.zeros((nframes, self.nat, 3))
        self.F = np.zeros((nframes, self.nat, 3))
        self.S = np.zeros((nframes, 3, 3))
        self.P = np.zeros((nframes, 3))
        self.q = np.zeros((nframes, self.nat, 3, 3))
        self.α = np.zeros((nframes, 3, 3))
        self.p = np.zeros((nframes, 3, 3))
        self.g = np.zeros((nframes, 3, 3))
        self.at = np.zeros((self.nat))

        for frame in range(nframes):

            # split 2nd line
            L = data[frame*(self.nat+2)+1]
            L = self.split(L,'=')
            L = self.split(L,' ')
            L = self.split(L,'"')
            L = L.split()

            # get indexes
            idx_latt = L.index('Lattice') + 1
            if 'energy' in L:
                idx_E = L.index('energy') + 1
            elif 'total_energy' in L:
                idx_E = L.index('total_energy') + 1
            idx_S = L.index('stress') + 1
            idx_P = L.index('polarization') + 1
            idx_α = L.index('polarizability') + 1

            # per frame quantities
            self.p[frame,:,:] = np.array([float(x) for x in L[idx_latt:idx_latt+9]]).reshape(3,3)
            self.g[frame,0,:] = self.p[frame,0,:]/np.linalg.norm(self.p[frame,0,:])
            self.g[frame,1,:] = self.p[frame,1,:]/np.linalg.norm(self.p[frame,1,:])
            self.g[frame,2,:] = self.p[frame,2,:]/np.linalg.norm(self.p[frame,2,:])
            self.E[frame]     = float(L[idx_E])
            self.S[frame,:,:] = np.array([ float(x) for x in L[idx_S:idx_S+9]]).reshape(3,3)
            self.P[frame,:]   = np.array([ float(x) for x in L[idx_P:idx_P+3]])
            self.α[frame,:,:] = np.array([ float(x) for x in L[idx_α:idx_α+9]]).reshape(3,3)

            # apply modulo polarization
            self.P[frame,:] = self.unit_pol(self.P[frame,:], self.g[frame,:,:], self.p[frame,:,:])

            # per atom quantities
            for iat in range(self.nat):
                L = data[frame*(self.nat+2)+2+iat].split()
                self.R[frame,iat,:]   = np.array([float(x) for x in L[1:4]])
                self.F[frame,iat,:]   = np.array([float(x) for x in L[4:7]])
                self.q[frame,iat,:,:] = np.array([float(x) for x in L[4:4+9]]).reshape(3,3)
                self.F[frame,iat,:]   = np.array([float(x) for x in L[13:13+3]])

    def unit_pol(self, P, g, p):
        pol_mod_frac = np.dot(np.linalg.inv(g), p).diagonal()
        P_frac = np.dot(g,P)  
        Pnew = P_frac % (np.sign(P_frac)*pol_mod_frac)
        Pnew = np.where(Pnew >  0.5*pol_mod_frac, Pnew - pol_mod_frac, Pnew)
        Pnew = np.where(Pnew < -0.5*pol_mod_frac, Pnew + pol_mod_frac, Pnew)
        Pnew = np.dot(np.linalg.inv(g), Pnew)
        return Pnew

def parity_plot(y1, y2, title, type, pdf):
    """
    y1: DFT     label
    y2: Allegro label
    type = 0: scatter plot
    type = 1: apply gaussian filter with arbitrary sigma
    type = 2: smoothened countour plot with default sigma
    """

    f = plt.figure(figsize=(6, 6), dpi=60)
    plt.rcParams.update({'font.size': 24})
    plt.gcf().subplots_adjust(left=0.15, bottom=0.19, top=0.79, right=0.95)
    ax = plt.gca()
    axis_settings(ax)
    plt.title(title)
    plt.xlabel('DFT')
    plt.ylabel('ML')

    mae = mean_absolute_error(y1, y2)
    rmse = np.sqrt(mean_squared_error(y1, y2))

    if type == 0:
        plt.scatter(y1,y2, s=10)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, '-', alpha=0.75, zorder=0, c='r', lw=2)
        plt.scatter(y1, y2, color='blue', alpha=0.6)

    if type == 1:
        N_bins = 100
        colors = [(1, 1, 1)]
        colors.extend(plt.cm.jet(i) for i in range(plt.cm.jet.N))
        cmap = LinearSegmentedColormap.from_list("custom_jet_white", colors, plt.cm.jet.N)
        hist, xedges, yedges = np.histogram2d(y1.ravel(), y2.ravel(), bins=N_bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(hist.T, extent=extent, origin='lower', cmap=cmap, interpolation='gaussian', aspect='auto', vmin=0, vmax=np.max(hist))
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])] 
        ax.plot(lims, lims, ':', alpha=0.30, c='r', lw=2)
        plt.colorbar()

    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    textstr = '\n'.join((
        r'mae  = %.4f' % (round(mae,4),),
        r'rmse = %.4f' % (round(rmse,4),)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.28, 0.82, textstr, transform=ax.transAxes, bbox=props, fontsize=16)

    pdf.savefig()

#--------------------------------------------------------------------------
#---------------------------- PRODUCTION ----------------------------------
#--------------------------------------------------------------------------

assert len(sys.argv) > 1, "Specify Material"

system = sys.argv[1]

if system == "SiO2":
    file_dft = "SiO2/SiO2.xyz"
    file_ml  = "SiO2/SiO2-ML.xyz"

if system == "BaTiO3":
    file_dft = "BaTiO3/BaTiO3.xyz"
    file_ml  = "BaTiO3/BaTiO3-ML.xyz"

else:
    print("Material not implemented")

S1 = extxyz(file_dft)
S2 = extxyz(file_ml)

with PdfPages(system+"/"+system+".pdf") as pdf:

    d = {0:"x",1:"y",2:"z"}

    # energy
    parity_plot(S1.E/S1.nat, S2.E/S2.nat, "$U/N$", type=1, pdf=pdf)

    # positions
    for i in range(3):
        parity_plot(S1.R[:,:,i], S2.R[:,:,i], "$R_{:s}$".format(d[i]), type=0, pdf=pdf)

    # forces
    for i in range(3):
        parity_plot(S1.F[:,:,i], S2.F[:,:,i], "$F_{:s}$".format(str(d[i])), type=1, pdf=pdf)

    # polarization
    for i in range(3):
        parity_plot(S1.P[:,i], S2.P[:,i], "$P_{:s}$".format(d[i]), type=1, pdf=pdf)

    # Born charges
    for i in range(3):
        for j in range(3):
            parity_plot(S1.q[:,i,j], S2.q[:,i,j], "$Z_{{{:s}{:s}}}$".format(str(d[i]),str(d[j])), type=1, pdf=pdf)

    # polarizability
    for i in range(3):
        for j in range(3):
            parity_plot(S1.α[:,i,j], S2.α[:,i,j],"$α_{{{:s}{:s}}}$".format(str(d[i]),str(d[j])), type=1, pdf=pdf)
