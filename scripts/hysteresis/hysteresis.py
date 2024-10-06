# =================
# HYSTERESIS LAMMPS
# =================

"""
Author: S. Falletta

This script parses the LAMMPS output file from an MLMD simulation at fixed volume 
(orthorhombic cells only) and temperature under a sinusoidal electric field. 
The simulation is performed using the pair-allegro interface, which includes 
the treatment of polarization, Born charges, and polarizability. When provided 
with multiple files, as in the case of finite-temperature MD, the polarization 
is averaged over all trajectories.

The script plots: 
• Polarization hysteresis vs electric field 
• Polarization vs time 
• Electric field vs time

How to run: python3 hysteresis.py BaTiO3-8640-T300

The input data must be specified at the bottom of this file.

Before using this script, ensure your folders are organized as in the provided example.

TODO: Generalize to non-orthorhombic cells.
"""

#Libraries
import sys
import numpy as np
import subprocess
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rcParams.update({'font.size': 24,'legend.fontsize': 14,'legend.handlelength': 0.5})
from matplotlib.backends.backend_pdf import PdfPages
import os

# colors
c_ml  = ["#0055d4",'#d40000']

# Constants
bohr2A = 0.529177249
A2bohr = 1.8897259886
hartree2eV = 27.211396641308
e2C  = 1.602176634e-19
A2m  = 1e-10
m2cm = 1e2
C2μC = 1e6
V2MV = 1e-6

def axis_settings(ax):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_tick_params(which='major', width=3.0, length=12, direction="in")
    ax.xaxis.set_tick_params(which='minor', width=3.0, length=6,  direction="in")
    ax.yaxis.set_tick_params(which='major', width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which='minor', width=3.0, length=6,  direction="in")
    ax.yaxis.set_ticks_position('both')

def plot_init(label_x, label_y, title):
    f = plt.figure(figsize=(6, 6), dpi=60)
    plt.gcf().subplots_adjust(left=0.19, bottom=0.19, top=0.79, right=0.99)
    ax = plt.gca()
    axis_settings(ax)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

class hysteresis_LAMMPS:
    def __init__(self, infile, units="SI"):

        """
        Units in the LAMMPS file
        • time in ps
        • volume in A^3
        • energy in eV
        • polarization density (in e/A^2)
        valid only for orthorombic cells. TODO: generalize to nonorthorombic cells
        """

        if not os.path.exists(infile):
            raise FileNotFoundError(f"The file '{infile}' does not exist.")

        self.units  = units
        self.infile = infile

        self.get_data()

    def get_data(self):

        # general stuff for orthorombic cells
        self.system = self.infile.split('.dat')[0]
        print("• "+self.system)

        data = subprocess.check_output("grep 'orthogonal box '  "+self.infile+" | tail -1", shell=True, text=True).split(' ')
        replica_line = subprocess.check_output("grep -A 1 'Replication is creating '  "+self.infile+" | tail -1", shell=True, text=True)
        if replica_line != "":
            data = subprocess.check_output("grep -A 1 'Replication is creating '  "+self.infile+" | tail -1", shell=True, text=True).split(' ')
        self.A = float(data[9].split('(')[1])
        self.B = float(data[10])
        self.C = float(data[11].split(')')[0])
        self.volume = self.A * self.B * self.C

        # data
        data = subprocess.check_output("grep -A 100000000 'v_efield         v_Pz'  "+self.infile+"", shell=True, text=True).splitlines() 
        data_vals = []
        for line in data:
            if "Step" in line:
                continue
            if "Loop time" in line:
                break
            data_vals.append(line)
        self.nframes = len(data_vals) 
        self.time = np.zeros(self.nframes)
        self.E    = np.zeros(self.nframes)
        self.P    = np.zeros(self.nframes)
        for i in range(self.nframes):
            self.E[i]    = float(data_vals[i].split()[6])
            self.P[i]    = float(data_vals[i].split()[7])
            self.time[i] = float(data_vals[i].split()[1])

        # eV units
        if self.units == "eV":
            self.Pconv = 1.0                                 # e * A
            self.Econv = 1.0                                 # V / A
            self.Ωconv = 1.0                                 # A^3
            self.Eunit = "V$\cdot$A$^{-1}$"
            self.Punit = "e$\cdot$A$^{-2}$"
        if self.units == "atomic":
            self.Pconv = np.sqrt(2.0) / bohr2A               # Ry a.u.
            self.Econv = (bohr2A * np.sqrt(2)) / hartree2eV  # Ry a.u.
            self.Ωconv = A2bohr**3                           # Bohr^3
            self.Eunit = "a.u."
            self.Punit = "Ry a.u. / Bohr$^3$"
        if self.units == "SI":
            self.Pconv = (e2C * C2μC) * (A2m * m2cm)         # μC * cm
            self.Econv = V2MV / (A2m * m2cm)                 # MV / cm
            self.Ωconv = (A2m * m2cm)**3                     # in cm^3
            self.Eunit = "MV$\cdot$cm$^{-1}$"
            self.Punit = "μC$\cdot$cm$^{-2}$"
        self.P *= self.Pconv / self.Ωconv
        self.E *= self.Econv

def plot_hysteresis_average(S, pdf, plot_time_ev):

    # Electric field and average polarization
    time = S[0].time
    E = S[0].E
    P = np.zeros(len(E))
    for i in range(len(infiles)):
        P += S[i].P
    P /= len(infiles)
    if len(S) > 1:
        print(f"Averaging polarization over {len(S)} trajectories")

    # plot average hysteresis
    plot_init("Efield ({:s})".format(S[0].Eunit), "P/$\Omega$ ({:s})".format(S[0].Punit),"Hysteresis")
    plt.plot(E, P, lw=2.5, color=c_ml[0], zorder=100)
    pdf.savefig()

    if plot_time_ev:
        # plot Efield
        plot_init("Time (ps)", "Efield ({:s})".format(S[0].Eunit), "Electric field")
        plt.plot(time, E, lw=2.5, color=c_ml[0], zorder=100)
        pdf.savefig()

        # plot polarization
        plot_init("Time (ps)", "P/$\Omega$ ({:s})".format(S[0].Punit), "Polarization")
        plt.plot(time, P, lw=2.5, color=c_ml[0], zorder=100)
        pdf.savefig()

#--------------------------------------------------------------------------
#---------------------------- PRODUCTION ----------------------------------
#--------------------------------------------------------------------------

assert len(sys.argv) > 1, "Specify Material"

system = sys.argv[1]

if system == "BaTiO3-8640-T0":
    infiles = ["BaTiO3-8640-T0.dat"]
    plot_time_ev = False

elif system == "BaTiO3-8640-T300":
    infiles = [f"BaTiO3-{i}.dat" for i in range(1, 11)]
    plot_time_ev = True
    
else:
    print(system+" not implemented")
    exit()

# Construct data
S = []
for infile in infiles:
    S.append(hysteresis_LAMMPS(system+"/"+infile))

# Plot data
with PdfPages(system+"/"+system+".pdf") as pdf:
    plot_hysteresis_average(S, pdf, plot_time_ev)
