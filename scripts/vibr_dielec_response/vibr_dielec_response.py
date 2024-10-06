# ==================================
# VIBRATIONAL AND DIELECRIC RESPONSE
# ==================================

"""
Author: S. Falletta

This script parses the LAMMPS outfile of a MLMD at fixed volume (orthorombic cell only)
fixed temperature, carried out with the pair-allegro interface that includes the 
treatment of polarization and polarizability, and plots:
• infrared spectrum
• real and imaginary parts of the frequency-dependent dielectric constant

How to run: python3 vibr_dielec_response.py SiO2

The input data needs to be specified at the bottom of this file.
DFPT results from QE can also be included, for comparison with the MLMD results.
The initial frequency ωi, and the final frequency ωf must be specified for the plots.

Before using this script, make sure your folders are organized as in this example.

TODO: Generalize to the case of non-orthorombic cells
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
plt.rcParams.update({"font.size": 24, "legend.fontsize": 16, "legend.handlelength": 0.5})
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
import sys
import tqdm
import subprocess
import os

# Constants and unit conversions
kB         = 8.617333262145e-5           # in eV K-1
hartree2eV = 27.211396641308
bohr2A     = 0.529177249
eps0const  = 5.5263499562 * 10**-3       # in [e * Volt^{-1} * Ansgrom^{-1}]
THz2cminv  = 0.03335640951981521 * 10**3
sig_g      = 20                          # gaussian broadening (in cm-1ƒ)
cm_inv2Ry  = 0.000124 * 13.605698320654
c_dfpt     = '#d40000'
c_mlmd     = "#0055d4"

# Extra settings
plot_smeared = True        # plot results with smearing included
freq_damp    = True        # damp non-vanishing real part of autocorr function at large frequency

#--------------------------------------------------------------------------
#------------------------- GENERIC FUNCTIONS ------------------------------
#--------------------------------------------------------------------------

def axis_settings(ax):
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.xaxis.set_tick_params(which="minor", width=3.0, length=6,  direction="in")
    ax.yaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which="minor", width=3.0, length=6,  direction="in")
    ax.yaxis.set_ticks_position("both")

def plot_init(label_x, label_y, title):
    f = plt.figure(figsize=(6, 6), dpi=60)
    plt.gcf().subplots_adjust(left=0.15, bottom=0.19, top=0.79, right=0.95)
    ax = plt.gca()
    axis_settings(ax)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

def ylim_range(data):
    ymax = max(data)
    ymin = min(data)
    d = (ymax-ymin)*0.15
    y1 = ymin-d
    y2 = ymax+d
    return y1,y2

def set_lims(x,xi,xf,y):
    return ylim_range(y[np.abs(x-xi).argmin():np.abs(x-xf).argmin()])

def rescale_amplitude(y1,y2,x1,x2,xi,xf):
    idx1i = np.abs(x1-xi).argmin()
    idx1f = np.abs(x1-xf).argmin()
    idx2i = np.abs(x2-xi).argmin()
    idx2f = np.abs(x2-xf).argmin()
    return y1 / np.nanmax(np.abs(y1[idx1i:idx1f])) * np.nanmax(np.abs(y2[idx2i:idx2f]))

def gaussian(x, A, mu, sig):
    return A/np.sqrt(2.0*np.pi*sig**2) * np.exp( -0.5 * ((x-mu)/sig)**2 )

def gaussian_kernel(x, sig_g):
    kernel = np.exp(-0.5 * (x / sig_g)** 2)
    return kernel / np.sum(kernel)

def gaussian_broaden(ω, data, sig_g, mode=0):
    """
    mode == 0: use np.convolve (symmetrize freq and signal)
    mode == 1: replace each point with its gaussian (not symmetric at zero freq, but same as in DFPT)
    """
    if mode == 0:
        ω_2    = np.concatenate((-ω[::-1][:-1],ω))
        data_2 = np.concatenate((data[::-1][:-1],data))
        kernel = gaussian_kernel(ω_2, sig_g)
        data_broadened = np.convolve(data_2, kernel, mode='same')[len(data)-1:len(ω_2)]
    if mode == 1:
        data_broadened = np.zeros(len(ω))
        for i,x in enumerate(data):
            gauss = gaussian(ω,1,ω[i],sig_g)
            data_broadened += x * gauss / np.sum(gauss)
    return data_broadened

#--------------------------------------------------------------------------
#------------- VIBRATIONAL AND DIELECTRIC PROPERTIES ----------------------
#--------------------------------------------------------------------------

class Spectroscopy:

    def __init__(self, file_mlmd, files_dfpt, temp):
        """
        file_mlmd:  LAMMPS log file
        files_dfpt: DFPT files for IR and freq-dependent dielectric constant
        do_dfpt:    flag for processing DFPT results from QE
        temp:       temperature
        """

        # General
        L = file_mlmd.split("/")
        self.pdfname = L[0]+"/"+L[-1][:-4]+".pdf"
        self.temp = temp

        # files from DFPT
        if files_dfpt == []:
            self.do_dfpt = False
        else:
            [file_IR_dfpt, file_ε_re_dfpt, file_ε_im_dfpt] = files_dfpt
            self.do_dfpt = True

        # Parse mlmd input
        self.parse_mlmd(file_mlmd)

        # calculate autocorrelation function of polarization
        self.get_autocorr_P()

        # calculate IR spectrum
        self.get_IR()

        # parse IR spectrum DFPT
        if self.do_dfpt:
            self.get_IR_dfpt(file_IR_dfpt)

        # determine high-frequency and static dielectric constants
        self.get_εinf_ε0()

        # calculate the frequency-dependent dielectric constant
        self.get_εω_mlmd()

        # parse the frequency-dependent dielectric constant from DFPT
        if self.do_dfpt:
            self.get_ε_dfpt(file_ε_re_dfpt, file_ε_im_dfpt)

        with PdfPages(self.pdfname) as pdf:

            # plot autocorrelation function of P from MDMD
            self.plot_autocorr_P(pdf)

            # plot IR spectra from mlmd and DFPT
            self.plot_IR(pdf)

            # plot dielectric constants
            self.plot_ε(pdf)

    def parse_mlmd(self, file_mlmd):
        """
        Units in the LAMMPS file
        * time in ps
        * volume in A**3
        * temperature in K
        * energy (in eV)
        * polarization (in e*A)
        * polarizability (in e*A^2*V^{-1})

        TODO: Extend the parser to non-orthogonal cells
        """

        # only for orthogonal cells
        data = subprocess.check_output("grep 'orthogonal box = (0 0 0)' "+file_mlmd+" | tail -1", shell=True, text=True).split()
        A = float(data[7].split('(')[1])
        B = float(data[8])
        C = float(data[9].split(')')[0])
        self.V = A * B * C

        # Parse total length production dynamics
        self.n_t = int(subprocess.check_output("grep 'run ' "+file_mlmd+" | tail -1 ", shell=True, text=True).split()[1]) + 1

        # Take the production dynamics
        with open(file_mlmd) as f:
            data = f.readlines()
        for i,line in enumerate(data):
            if 'c_polarization[1]' in line:
                i_start = i+1
        data = data[i_start:i_start+self.n_t]

        # Parse the production dynamics
        self.time = np.zeros(self.n_t)
        self.T    = np.zeros(self.n_t)
        self.E    = np.zeros(self.n_t)
        self.P    = np.zeros((self.n_t,3))
        self.α    = np.zeros((self.n_t,3,3))
        for i,line in enumerate(data):
            L = [float(x) for x in line.split()]
            self.time[i]  = L[1]
            self.T[i]     = L[2]
            self.E[i]     = L[4]
            self.P[i,:]   = L[6:9] 
            self.α[i,:,:] = np.array(L[9:18]).reshape((3,3))

        self.time -= self.time[0]

        # Useful quantities
        self.n_ω = int(self.n_t/2) + 1   # rfft gives half vector size
        self.var_P = np.var(self.P, axis=0, dtype=np.float64)
        self.var_α = np.array([np.var(self.α[:,i,i], dtype=np.float64) for i in range(3)])

    def get_autocorr_P(self):
        """
        Get autocorrelation function of polarization in time and frequency space from MLMD
        """
        self.P_AF_t = np.zeros((self.n_t,3))
        self.P_AF_ω = np.zeros((self.n_ω,3), dtype=complex)
        for i in range(3):
            self.ω, self.P_AF_t[:,i], self.P_AF_ω[:,i] = self.get_autocorr_mlmd(self.time, self.P[:,i])

            # fix for non-vanishing real part at large frequency
            if freq_damp:
                imag_part = np.array(self.P_AF_ω[:,i].imag)
                real_part = np.array(self.P_AF_ω[:,i].real)
                vmax = np.max(real_part)
                for j in range(self.n_ω):
                    if abs(real_part[j]) < 0.01*vmax:
                        real_part[j] = 0
                self.P_AF_ω[:,i] = real_part + 1.0j * imag_part

        self.P_AF_t_av = np.sum(self.P_AF_t, axis=1) / sum(self.var_P)
        self.P_AF_ω_av = np.sum(self.P_AF_ω, axis=1) / sum(self.var_P) * (-1)

    def get_autocorr_α(self):
        """
        Get autocorrelation function of polarizability in time and frequency space from MLMD.
        Only diagonal components, so vector of dimension 3
        """
        self.α_AF_t = np.zeros((self.n_t,3))
        self.α_AF_ω = np.zeros((self.n_ω,3), dtype=complex)
        for i in range(3):
            self.ω, self.α_AF_t[:,i], self.α_AF_ω[:,i] = self.get_autocorr_mlmd(self.time, self.α[:,i,i])
        self.α_AF_t_av = np.sum(self.α_AF_t, axis=1) / sum(self.var_α)
        self.α_AF_ω_av = np.sum(self.α_AF_ω, axis=1) / sum(self.var_α)

    def get_autocorr_mlmd(self,t, v, m=1):
        """
        calculate the autocorrelation function phi of a vector v
        averaging over m ensembles

        OSS:
        1) np.correlated is not normalized and must be divided by n
        2) the integral ∫dt becomes  Σ dt, where dt = T / N = t[1]-t[0]
        dt is expressed in units of ps (ps and THz are consistent)
        """

        n = len(v)
        dt = (t[1]-t[0]) / THz2cminv # in cm units

        # consider m different final points
        ω = np.fft.rfftfreq(n,t[1]-t[0]) * THz2cminv
        AF_t = np.zeros(n)
        AF_ω = None
        # calculate the autocorrelation function for each different final point
        for i in tqdm.trange(1,m+1):
            # construct a signal stopping at t_i
            v_aux = np.zeros(n)
            v_aux[:int(i*n/m)] = v[:int(i*n/m)]
            # calculate its autocorrelation function in time space
            AF_t_2 = np.correlate(v_aux, v_aux, mode='full') / (int(i*n/m))
            # take the second half of values
            AF_t_m = AF_t_2[n-1:]
            # calculate its fourier transform
            AF_ω_m = np.fft.rfft(AF_t_m) * dt
            # add them to the total
            AF_t += AF_t_m
            if AF_ω is None:
                AF_ω = AF_ω_m
            else:
                AF_ω += AF_ω_m
        # divide the result by the total number of ensembles
        AF_t /= m
        AF_ω /= m

        # To match results from Kramer's Kronig relations (<-> Parseval)
        AF_ω *= np.pi

        return ω, AF_t, AF_ω

    def get_IR(self):
        """
        get infrared spectrum from autocorrelation function of polarization
        """
        self.IR = self.P_AF_ω_av.real / sum(self.var_P) * self.ω**2

    def get_IR_dfpt(self,file):
        """
        parse IR DFPT peaks and construct the IR response
        """
        # extract peaks
        data = np.genfromtxt(file, skip_header=1)
        ω_peaks_dfpt  = data[:,1]
        IR_peaks_dfpt = data[:,3]

        # broaden peaks with gaussians
        self.IR_dfpt = np.zeros(self.n_ω)
        for mean, strength in zip(ω_peaks_dfpt, IR_peaks_dfpt):
            self.IR_dfpt += gaussian(self.ω, strength, mean, sig_g)
        self.IR_dfpt *= -1

    def get_εinf_ε0(self):
        """
        determine high-frequency dielectric constants from polarizability
        and static dielectric constant through the fluctuation-dissipation theorem
        """
        self.εinf = np.zeros(3)
        self.ε0   = np.zeros(3)
        for i in range(3):
            self.εinf[i] = 1 + 1.0/(eps0const*self.V) * np.sum(self.α[:,i,i])/self.n_t
            self.ε0[i]   = self.εinf[i] + self.var_P[i]/(kB * self.temp * self.V * eps0const)
        print("High-freq dielectric constant: {:.4f}".format(round(np.sum(self.εinf)/3,4)))
        print("Static    dielectric constant: {:.4f}".format(round(np.sum(self.ε0)/3,4)))

    def get_εω_mlmd(self):
        """
        frequency-depedent dielectric constant during the mlmd
        """
        # frequency dependent dielectric constant
        ε_ω = np.zeros((self.n_ω,3),dtype=complex)
        for i in range(3):
            ε_ω[:,i] = 1 + (self.ε0[i] - 1) * (1 - 1j * self.ω[:]*self.P_AF_ω[:,i] / self.var_P[i])

        # average real and imaginary part
        self.ε_ω_av_re = np.sum(ε_ω.real, axis=1) / 3
        self.ε_ω_av_im = np.sum(ε_ω.imag, axis=1) / 3

    def get_ε_dfpt(self, file_ε_re_dfpt, file_ε_im_dfpt):
        """
        parse real and imaginary part of the dielectric constant from DFPT
        """
        # real part
        data = np.genfromtxt(file_ε_re_dfpt)
        self.ω_ε_dfpt = data[:,0]
        self.ε_re_dfpt = data[:,1]
        print("Static (DFPT):                 {:.4f}".format(round(self.ε_re_dfpt[0],4)))

        # imaginary part
        data = np.genfromtxt(file_ε_im_dfpt)
        self.ε_im_dfpt = data[:,1]

    def plot_autocorr_P(self, pdf):
        """
        Plot average autocorrelation function of polarization in time and frequency space from MLMD
        """
        # plot the AF in time space
        plot_init("$t$ (ps)", "", r"$\frac{\langle P(t)\cdot P(0)\rangle}{var(P)}$")
        plt.plot(self.time, self.P_AF_t_av, lw=2.5, c=c_mlmd)
        pdf.savefig()

    def plot_IR(self, pdf, do_smear=True):
        """
        Plot infrared spectrum
        """
        # rescale DFPT result
        plot_init("$\omega$ (cm$^{-1}$)","Infrared Intensity", "")
        plt.xlim(ωi,ωf)
        ax = plt.gca()
        ax.axes.yaxis.set_ticks([])
        if do_smear is False:
            plt.plot(self.ω, self.IR, lw=2.5, label="MLMD", c=c_mlmd)
            plt.ylim(set_lims(self.ω,ωi,ωf,self.IR))
        else:
            IR_gauss = gaussian_broaden(self.ω, self.IR, sig_g)
            plt.plot(self.ω, IR_gauss, lw=2.5, c=c_mlmd, label='MLMD')
            plt.ylim(set_lims(self.ω,ωi,ωf,IR_gauss))

        if self.do_dfpt:
            if do_smear is False:
                IR_dfpt = rescale_amplitude(self.IR_dfpt,self.IR,self.ω,self.ω,ωi,ωf)
            else:
                IR_dfpt = rescale_amplitude(self.IR_dfpt,IR_gauss,self.ω,self.ω,ωi,ωf)
            plt.plot(self.ω, IR_dfpt, lw=4, label="DFPT", ls=":", c=c_dfpt)

        plt.legend(frameon=False)
        pdf.savefig()

    def plot_ε(self, pdf, do_smear=True):
        """
        Plot MLMD and DFPT dielectric constants
        """

        # real part
        plot_init(r"$\omega$ (cm$^{-1}$)", "Re$[ε]$", "")
        plt.xlim(ωi,ωf)
        if do_smear is False:
            plt.plot(self.ω, self.ε_ω_av_re, lw=2.5, c=c_mlmd, label='MLMD')
            plt.ylim(set_lims(self.ω,ωi,ωf,self.ε_ω_av_re))
        else:
            ε_ω_av_re_gauss = gaussian_broaden(self.ω, self.ε_ω_av_re, sig_g)
            plt.plot(self.ω, ε_ω_av_re_gauss, lw=2.5, c=c_mlmd, label='MLMD')
        if self.do_dfpt:
            plt.plot(self.ω_ε_dfpt, self.ε_re_dfpt, lw=4, c=c_dfpt, ls=":", label='DFPT')
        plt.legend(frameon=False)
        plt.ylim(-8,14)
        pdf.savefig()

        # imaginary part
        plot_init(r"$\omega$ (cm$^{-1}$)", "$-$Im$[ε]$", "")
        plt.xlim(ωi,ωf)
        if do_smear is False:
            plt.plot(self.ω, -self.ε_ω_av_im, lw=2.5, c=c_mlmd, label='MLMD')
            plt.ylim(set_lims(self.ω,ωi,ωf,self.ε_ω_av_im))
        else:
            ε_ω_av_im_gauss = gaussian_broaden(self.ω, self.ε_ω_av_im, sig_g)
            plt.plot(self.ω, -ε_ω_av_im_gauss, lw=2.5, c=c_mlmd, label='MLMD')
        if self.do_dfpt:
            plt.plot(self.ω_ε_dfpt, self.ε_im_dfpt, lw=4, c=c_dfpt, ls=":", label='DFPT')
        plt.legend(frameon=False)
        pdf.savefig()

#--------------------------------------------------------------------------
#----------------------------- PRODUCTION ---------------------------------
#--------------------------------------------------------------------------

assert len(sys.argv) > 1, "Specify Material"

system = sys.argv[1]

if system == "SiO2":
    files_mlmd = ["SiO2/ML/SiO2-mlmd.dat"]
    #files_dfpt = [] # leave it empty if you don't have the QE results
    files_dfpt = ["SiO2/DFPT/SiO2-IR-dfpt.dat", "SiO2/DFPT/SiO2-epsre-dfpt.dat", "SiO2/DFPT/SiO2-epsim-dfpt.dat"] # calculated with QE
    ωi = 0; ωf = 1400
    temp = 300

else:
    print("Material not implemented")
    exit()

for file_mlmd in files_mlmd:
    # check if file exists
    print(file_mlmd)
    if os.path.exists(file_mlmd) is False:
        print("File {:s} does not exist!".format(file_mlmd))

    # processing the file
    print("\nProcessing "+file_mlmd)
    Spectroscopy(file_mlmd, files_dfpt, temp)
