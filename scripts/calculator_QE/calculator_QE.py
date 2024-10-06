# ======================================
# QUANTUM ESPRESSO INPUT GENERATOR CLASS
# ======================================

"""
Author: S. Falletta

This script generates the input files for performing the QE calculations for
multiple trajectory files under zero electric field, and small finite electric
field along x, y, and z. The trajectory files must be in the format extxyz.

The class QEGenerator takes care of composing the input file.
Each material needs to be included with its own class.
We here provide the classes for SiO2 and BaTiO3.
"""

# Library
import numpy as np
import sys
import os

# directories in the cluster
pseudo_dir    = '/n/holyscratch01/kozinsky_lab/sfalletta/QE/Inputs'
outdir_prefix = '/n/holyscratch01/kozinsky_lab/sfalletta/QE/'

#--------------------------------------------------------------------------
#---------------------------- QE GENERATOR---------------------------------
#--------------------------------------------------------------------------

class QEGenerator:

    def __init__(self, output_file, xyzfile, prefix, pseudo_dir, outdir):
        """
        Initialize the QEGenerator class with default paths and file names.

        Parameters:
        - output_file: str, path to save the generated input file.
        - xyzfile:     str, path to the XYZ coordinate file.
        - pseudo_dir:  str, path to the pseudopotential directory.
        - outdir:      str, output directory for Quantum Espresso calculations.
        """
        self.output_file = output_file
        self.xyzfile     = xyzfile

        # Default values for parameters
        self.control_params = {
            'calculation':  'scf',
            'restart_mode': 'from_scratch',
            'prefix':        prefix,
            'pseudo_dir':    pseudo_dir,
            'outdir':        outdir,
            'etot_conv_thr': 1.0e-10,
            'forc_conv_thr': 1.0e-8,
            'tstress':       '.true.',
            'tprnfor':       '.true.'
        }
        self.system_params = {
            'input_dft':   'PBE',
            'ecutwfc':      100,
            'ecutrho':      800,
            'ibrav':        8,
            'occupations': 'fixed',
            'nbnd':         550
        }
        self.electron_params  = {
            'conv_thr':         1.0e-8,
            'electron_maxstep': 500,
            'mixing_beta':      0.7,
            'diagonalization': 'david',
            'diago_full_acc':  '.true.',
        }

        self.atomic_species   = []
        self.k_points         = []
        self.atomic_positions = []
        self.atoms            = []

    def set_control_params(self, **kwargs):
        """Update CONTROL parameters."""
        self.control_params.update(kwargs)

    def set_system_params(self, **kwargs):
        """Update SYSTEM parameters."""
        self.system_params.update(kwargs)

    def set_electron_params(self, **kwargs):
        """Update ELECTRONS parameters."""
        self.electron_params.update(kwargs)

    def set_atomic_species(self, species_list):
        """Set ATOMIC_SPECIES (list of tuples)."""
        self.atomic_species = species_list

    def set_k_points(self, k_points):
        """Set K_POINTS (either list or string)."""
        self.k_points = k_points

    def set_atomic_positions(self):
        """Set ATOMIC_POSITIONS and set structural SYSTEM parameters"""
        with open(self.xyzfile, 'r') as fxyz:
            lines = fxyz.readlines()
        kinds = set()
        for line in lines[2:]:
            L = line.split()
            self.atoms.append([L[0],float(L[1]),float(L[2]),float(L[3])])
            kinds.add(L[0])

        # set SYSTEM parameters
        self.set_system_params(
            nat=int(lines[0]),
            ntyp=len(kinds)
        )

        # extract lattice parameters
        if "Lattice" not in lines[1]:
            print("'{:s}' must be formatted in extxyz!".format(infile))
            exit()
        p = np.array([float(x) for x in lines[1].split('"')[1].split()]).reshape((3,3))
        A = np.linalg.norm(p[0,:])
        B = np.linalg.norm(p[1,:])
        C = np.linalg.norm(p[2,:])
        cosBC = np.dot(p[1,:], p[2,:]) / (B*C)
        cosAC = np.dot(p[0,:], p[2,:]) / (A*C)
        cosAB = np.dot(p[0,:], p[1,:]) / (A*B)

        self.set_system_params(
            A = A,
            B = B,
            C = C,
            cosBC = cosBC,
            cosAC = cosAC,
            cosAB = cosAB,
        )

    def write_qe_input(self):
        """
        Generate the Quantum Espresso input file with the current class parameters.
        """
        def format_param(key, value):
            """Format the parameters so that values are aligned neatly."""
            key_padding = 18  # Adjust for consistent alignment (number of spaces)
            value_str = repr(value)
            return f"    {key:<{key_padding}}= {value_str}\n"

        with open(self.output_file, 'w') as f:

            # Write CONTROL section
            f.write("&CONTROL\n")
            for key, value in self.control_params.items():
                f.write(format_param(key, value))
            f.write("/\n\n")

            # Write SYSTEM section
            f.write("&SYSTEM\n")
            for key, value in self.system_params.items():
                f.write(format_param(key, value))
            f.write("/\n\n")

            # Write ELECTRONS section
            f.write("&ELECTRONS\n")
            for key, value in self.electron_params.items():
                f.write(format_param(key, value))
            f.write("/\n\n")

            # Write ATOMIC_SPECIES section
            f.write("ATOMIC_SPECIES\n")
            for species in self.atomic_species:
                element, mass, pseudo = species
                f.write(f"    {element:<10} {mass:<8} {pseudo}\n")
            f.write("\n")

            # Write K_POINTS section
            f.write("K_POINTS automatic\n")
            if isinstance(self.k_points, list):
                k_points_str = " ".join(map(str, self.k_points))
                f.write(f"    {k_points_str}\n")
            else:
                f.write(f"    {self.k_points}\n")
            f.write("\n")

            # Write ATOMIC_POSITIONS section
            f.write('ATOMIC_POSITIONS  angstrom\n')
            for atom in self.atoms:
                str_format="{:5s} "+"{:14.10f} "*3+" \n"
                f.write(str_format.format(atom[0],atom[1], atom[2], atom[3])) 


#--------------------------------------------------------------------------
#-------------------------- CLASS MATERIALS -------------------------------
#--------------------------------------------------------------------------

class SiO2:

    def __init__(self, xyzfile, pseudo_dir, outdir, dir, efield=[0,0,0]):

        prefix = "SiO2"
        output_file = f"{dir}/{prefix}-scf.inp"

        # Initialize the generator and set parameters
        self.S = QEGenerator(output_file=output_file, xyzfile=xyzfile, prefix=prefix, pseudo_dir=pseudo_dir, outdir=outdir)
        self.S.set_control_params(
            lelfield='.true.',
            nberrycyc=1,
        )
        self.S.set_system_params(
            ibrav=14,
            nbnd=220
        )
        self.S.set_electron_params(
            efield_cart_1=efield[0],
            efield_cart_2=efield[1],
            efield_cart_3=efield[2]
        )
        self.S.set_atomic_species([
            ('Si', 28.0855, 'Si.upf'),
            ('O',  15.999,  'O.upf')
        ])
        self.S.set_k_points([2, 2, 2, 0, 0, 0])
        self.S.set_atomic_positions()
        self.S.write_qe_input()

class BaTiO3:

    def __init__(self, xyzfile, pseudo_dir, outdir, dir, efield=[0,0,0]):

        prefix = "BaTiO3"
        output_file = f"{dir}/{prefix}-scf.inp"

        # Initialize the generator and set parameters
        self.S = QEGenerator(output_file=output_file, xyzfile=xyzfile, prefix=prefix, pseudo_dir=pseudo_dir, outdir=outdir)
        self.S.set_control_params(
            lelfield='.true.',
            nberrycyc=1,
        )
        self.S.set_system_params(
            ibrav=8,
            nbnd=550
        )
        self.S.set_electron_params(
            efield_cart_1=efield[0],
            efield_cart_2=efield[1],
            efield_cart_3=efield[2]
        )
        self.S.set_atomic_species([
            ('Ba', 137.327, 'Ba.upf'),
            ('Ti', 47.90,   'Ti.upf'),
            ('O',  15.999,  'O.upf')
        ])
        self.S.set_k_points([1, 1, 1, 0, 0, 0])
        self.S.set_atomic_positions()
        self.S.write_qe_input()

#--------------------------------------------------------------------------
#----------------------------- PRODUCTION ---------------------------------
#--------------------------------------------------------------------------

assert len(sys.argv) > 1, "Specify Material"

system = sys.argv[1]

if system == "SiO2":

    traj_files = ["SiO2/SiO2-T300.xyz","SiO2/SiO2-T600.xyz"]

    # loop over the trajectory files
    for traj in traj_files:

        # print the extxyz file
        with open(traj, 'r') as f:
            lines = f.readlines()
        nat = int(lines[0])
        nframes = int(len(lines)/(nat+2))

        # loop over the frames of a given trajectory file
        for frame in range(nframes):

            # write the extxyz file
            xyzfile = f"{system}/{system}-sc333-{frame}.xyz"
            with open(xyzfile, 'w') as f:
                for i in range(nat+2):
                    f.write(lines[frame*(nat+2)+i])

            # loop over the electric fields
            for i, E in enumerate(["0","x","y","z"]):

                # define directory and the QE key outdir
                dir = f"{traj[:-4]}-E{E}-{frame}"
                outdir  = outdir_prefix+dir

                # efield
                efield = [0.0, 0.0, 0.0]
                if i > 0:
                    efield[i-1] = 1e-4

                # generate folder with input
                os.makedirs(dir, exist_ok=True)
                S = SiO2(xyzfile=xyzfile, pseudo_dir=pseudo_dir, outdir=outdir, dir=dir, efield=efield)

            os.remove(xyzfile)

elif system == "BaTiO3":

    traj_files = ["BaTiO3/BaTiO3.xyz","BaTiO3/BaTiO3-wall.xyz"]

    # loop over the trajectory files
    for traj in traj_files:

        # print the extxyz file 
        with open(traj, 'r') as f:
            lines = f.readlines()
        nat = int(lines[0])
        nframes = int(len(lines)/(nat+2))

        # loop over the frames of a given trajectory file
        for frame in range(nframes):

            # write the extxyz file
            xyzfile = f"{system}/{system}-sc333-{frame}.xyz"
            with open(xyzfile, 'w') as f:
                for i in range(nat+2):
                    f.write(lines[frame*(nat+2)+i])

            # loop over the electric fields
            for i, E in enumerate(["0","x","y","z"]):

                # define directory and the QE key outdir
                dir = f"{traj[:-4]}-E{E}-{frame}"
                outdir  = outdir_prefix+dir

                # efield
                efield = [0.0, 0.0, 0.0]
                if i > 0:
                    efield[i-1] = 1e-4

                # generate input
                os.makedirs(dir, exist_ok=True)
                S = BaTiO3(xyzfile=xyzfile, pseudo_dir=pseudo_dir, outdir=outdir, dir=dir, efield=efield)

            os.remove(xyzfile)

else:
    print("Material not implemented")
    exit()
