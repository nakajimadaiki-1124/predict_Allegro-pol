#===========================
# STATIC DIELECTRIC CONSTANT
#===========================

"""
Author: S. Falletta

This script parses:
• QuantumEspresso output file of a structural relaxation under an electric field
• LAMMPS output file of a structural relaxation carried out with the pair-allegro interface, 
  including the treatment of polarization and polarizability

It then plots the dielectric constant throughout the relaxation process.

How to run: python3 dielectric_constant.py SiO2

The input data needs to be specified at the bottom of this file.

TODO: generalize for non-orthorhombic cells.
"""

#Libraries
import sys
import numpy as np
import subprocess
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import (AutoMinorLocator)
plt.rcParams.update({'font.size': 24,'legend.fontsize': 20,'legend.handlelength': 0.5})
from matplotlib.backends.backend_pdf import PdfPages
import os

# colors
c_dft = '#d40000'
c_ml  = "#0055d4"

# Constants and unit conversions
bohr2A     = 0.529177249
hartree2eV = 27.211396641308
eps0const  = 5.5263499562 * 10**-3   # in [e * Volt^{-1} * Ansgrom^{-1}]
Punit      = bohr2A / np.sqrt(2.0)
Efieldunit = hartree2eV / (bohr2A * np.sqrt(2))
Vunit      = bohr2A**3

#--------------------------------------------------------------------------
#-------------------------- GENERIC FUNCTIONS -----------------------------
#--------------------------------------------------------------------------

def axis_settings(ax):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_tick_params(which='major', width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which='major', width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which='minor', width=3.0, length=6, direction="in")
    ax.yaxis.set_ticks_position('both')

#--------------------------------------------------------------------------
#----------------------- DIELECTRICS CONSTANT -----------------------------
#--------------------------------------------------------------------------

def plot_epsilon(P, P0, E, V, xlabel, title, col):
    """
    Arrays of:
        P:  polarization along z in the presence of the field
        P0: polarization along z in the absence of the field
        E:  electric field along z
        V:  volume
    """
    N = len(P)
    f = plt.figure(figsize=(6,6),dpi=60)
    plt.gcf().subplots_adjust(left=0.15,bottom=0.19,top=0.79,right=0.95)
    ax = plt.gca()
    axis_settings(ax)
    plt.xlabel(xlabel)
    plt.ylabel("ε")
    plt.title(title)
    for i in range(N):
        t = range(len(P[i]))
        ε = np.array([1 + (P[i][n]-P0[i][0]) / (E[i] * eps0const * V[i]) for n in t])
        plt.scatter(t, ε, lw=5, c=col)
        plt.plot(t, ε, lw=2.5, c=col)
        plt.scatter(t[0],  ε[0],  c='C2', s=125, zorder=100)
        plt.scatter(t[-1], ε[-1], c='C2', s=125, zorder=100)
        ax.text(t[-1]*0.2, ε[0]*1.0,   "ε$_∞$ = "+str(round(ε[0],2)),  ha='center', va='center', color='C2', fontsize=20)
        ax.text(t[-1]*0.8, ε[-1]*0.95, "ε$_0$ = "+str(round(ε[-1],2)), ha='center', va='center', color='C2', fontsize=20)
    pdf.savefig()

#--------------------------------------------------------------------------
#------------------------------- PARSERS ----------------------------------
#--------------------------------------------------------------------------

class LAMMPS_outfile:
    def __init__(self,infile,zerofield=False):
        """
        Parse polarization values during the LAMMPS relaxation
        Units in the LAMMPS file
        * time in ps
        * volume in A**3
        * polarization (in e*A)
        Valid only for orthorombic cells
        """

        if not os.path.exists(infile):
            raise FileNotFoundError(f"The file '{infile}' does not exist.")

        # general info
        self.system = infile.split('.dat')[0]
        print("• "+self.system)
        data = subprocess.check_output("grep -A 1 'Replication is creating a 1x1x1'  "+infile+" | tail -1", shell=True, text=True).split(' ')
        self.A = float(data[9].split('(')[1])
        self.B = float(data[10])
        self.C = float(data[11].split(')')[0])
        data = subprocess.check_output("grep 'fix born all addbornforce'  "+infile+" | tail -1", shell=True, text=True).split()[-3:]
        self.Efield = np.array([float(x) for x in data])
        if zerofield:
            self.Efield = np.zeros(3)
        self.volume = self.A * self.B * self.C
        self.nat = int(subprocess.check_output("grep 'atoms' "+infile+" | head -2 | tail -1", shell=True, text=True).splitlines()[0].split()[0])

        # Polarization
        data = subprocess.check_output("grep -A 100000000 'Step          Time'  "+infile+"", shell=True, text=True).splitlines()
        data_vals = []
        for line in data:
            if "Step" in line:
                continue
            if "Loop time" in line:
                break
            data_vals.append(line)
        self.nframes = len(data_vals)
        self.time = np.zeros(self.nframes)
        self.P    = np.zeros((self.nframes,3))
        for i in range(self.nframes):
            self.time[i] = float(data_vals[i].split()[1])
            self.P[i,:]  = np.array([float(x) for x in data_vals[i].split()[6:9]])
            # add contributions due to polarizability if the field is non zero
            if np.abs(np.sum(self.Efield)) > 0:
                alpha = np.array([float(x) for x in data_vals[i].split()[12:12+9]]).reshape(3,3)
                self.P[i,:] += np.dot(self.Efield, alpha)

class QE_outfile:
    """
    Parse polarization values during the DFT relaxation
    """

    def __init__(self,infile):

        # general info
        self.system = infile.split('.out')[0]
        print("• "+self.system)
        with open(infile, 'r') as file:
            data = file.read()
        self.nframes = data.count("End of self-consistent calculation")
        self.volume = float(subprocess.check_output("grep 'unit-cell volume' "+infile, shell=True, text=True).split()[3]) * Vunit
        string="In a.u.(Ry)  cartesian system of reference"
        Efield = subprocess.check_output("grep -A 3 '"+string+"' "+infile, shell=True, text=True).splitlines()[1:]
        self.Efield = np.array([float(x) for x in Efield]) * Efieldunit

        # Polarization
        self.P = np.zeros((self.nframes,3))
        string="End of self-consistent calculation"
        data = subprocess.check_output("grep -B 10 '"+string+"' "+infile, shell=True, text=True).splitlines()
        for n in range(self.nframes):
            P_el  = np.array([float(data[j+n*12].split()[1]) for j in range(3)  ])*Punit
            P_ion = np.array([float(data[j+n*12].split()[1]) for j in range(4,7)])*Punit
            self.P[n,:] = P_ion + P_el

#--------------------------------------------------------------------------
#---------------------------- PRODUCTION ----------------------------------
#--------------------------------------------------------------------------

assert len(sys.argv) > 1, "Specify Material"

system = sys.argv[1]

if system == "SiO2":
    data_DFT_E0 = 'SiO2/DFT/SiO2-E0.out'
    data_DFT_Ez = 'SiO2/DFT/SiO2-Ez-1e-3.out'
    data_ML     = 'SiO2/ML/SiO2-sc222_1e-3.dat'
    ix = 2        # direction efield (0:x, 1:y, 2:z)
    idx_ML_f = 21 # index at which to truncate the ML results

else:
    print("Material not implemented")
    exit()

# Parse files
S0_DFT = QE_outfile(data_DFT_E0)
Sz_DFT = QE_outfile(data_DFT_Ez)
S0_ML  = LAMMPS_outfile(data_ML, zerofield=True)
Sz_ML  = LAMMPS_outfile(data_ML)

with PdfPages(system+"/"+system+".pdf") as pdf:

    # Dielectric constant from QE trajectory
    plot_epsilon([Sz_DFT.P[:,ix]], [S0_DFT.P[:,ix]], [Sz_DFT.Efield[ix]], [Sz_DFT.volume], "relax steps", "DFT", col=c_dft)

    # Dielectric constant from LAMMPS trajectory
    plot_epsilon([Sz_ML.P[:idx_ML_f,ix]],[S0_ML.P[:idx_ML_f,ix]],[Sz_ML.Efield[ix]],[Sz_ML.volume], "relax steps", "LAMMPS", col=c_ml)
