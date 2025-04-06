"""
QE input generator for calculations under zero and finite electric fields.
Author: S. Falletta

Generates QE input files for trajectory files under zero field and small 
finite electric fields along x, y, and z. Requires extxyz format.
"""

import numpy as np
import sys
import os

# Cluster directories
PSEUDO_DIR = './'
OUTDIR = './'


# Material configurations
MATERIAL_CONFIG = {
    'SiO2': {
        'ibrav': 14,
        'nbnd': 220,
        'species': [
            ('Si', 28.0855, 'Si.upf'),
            ('O', 15.999, 'O.upf')
        ],
        'k_points': [2, 2, 2, 0, 0, 0],
        'efield_magnitude': 1e-4
    },
    'BaTiO3': {
        'ibrav': 8,
        'nbnd': 550,
        'species': [
            ('Ba', 137.327, 'Ba.upf'),
            ('Ti', 47.90, 'Ti.upf'),
            ('O', 15.999, 'O.upf')
        ],
        'k_points': [1, 1, 1, 0, 0, 0],
        'efield_magnitude': 1e-4
    }
}

class QEGenerator:
    def __init__(
        self, output_file, prefix, pseudo_dir, outdir, frame_data=None
    ):
        self.output_file = output_file
        self.frame_data = frame_data
        
        self.control_params = {
            'calculation': 'scf',
            'restart_mode': 'from_scratch',
            'prefix': prefix,
            'pseudo_dir': pseudo_dir,
            'outdir': outdir,
            'etot_conv_thr': 1.0e-10,
            'forc_conv_thr': 1.0e-8,
            'tstress': '.true.',
            'tprnfor': '.true.'
        }
        
        self.system_params = {
            'input_dft': 'PBE',
            'ecutwfc': 100,
            'ecutrho': 800,
            'ibrav': 8,
            'occupations': 'fixed',
            'nbnd': 550
        }
        
        self.electron_params = {
            'conv_thr': 1.0e-8,
            'electron_maxstep': 500,
            'mixing_beta': 0.7,
            'diagonalization': 'david',
            'diago_full_acc': '.true.'
        }
        
        self.atomic_species = []
        self.k_points = []
        self.atomic_positions = []
        self.atoms = []

    def set_params(self, param_dict, **kwargs):
        param_dict.update(kwargs)

    def set_atomic_species(self, species_list):
        self.atomic_species = species_list

    def set_k_points(self, k_points):
        self.k_points = k_points

    def set_atomic_positions(self):
        if not self.frame_data:
            raise ValueError("No frame data provided")
            
        lines = self.frame_data.splitlines()
        kinds = set()
        for line in lines[2:]:
            L = line.split()
            self.atoms.append([L[0], float(L[1]), float(L[2]), float(L[3])])
            kinds.add(L[0])

        self.set_params(self.system_params, nat=int(lines[0]), ntyp=len(kinds))

        if "Lattice" not in lines[1]:
            raise ValueError("Frame data must be formatted in extxyz!")
            
        p = np.array([float(x) for x in lines[1].split('"')[1].split()]).reshape((3, 3))
        A, B, C = [np.linalg.norm(p[i, :]) for i in range(3)]
        cosBC = np.dot(p[1, :], p[2, :]) / (B * C)
        cosAC = np.dot(p[0, :], p[2, :]) / (A * C)
        cosAB = np.dot(p[0, :], p[1, :]) / (A * B)

        self.set_params(
            self.system_params, 
            A=A, B=B, C=C, 
            cosBC=cosBC, cosAC=cosAC, cosAB=cosAB
        )

    def write_qe_input(self):
        def format_param(key, value):
            return f"    {key:<18}= {repr(value)}\n"

        with open(self.output_file, 'w') as f:
            for section, params in [
                ('CONTROL', self.control_params),
                ('SYSTEM', self.system_params),
                ('ELECTRONS', self.electron_params)
            ]:
                f.write(f"&{section}\n")
                for key, value in params.items():
                    f.write(format_param(key, value))
                f.write("/\n\n")

            f.write("ATOMIC_SPECIES\n")
            for element, mass, pseudo in self.atomic_species:
                f.write(f"    {element:<10} {mass:<8} {pseudo}\n")
            f.write("\n")

            f.write("K_POINTS automatic\n")
            k_points_str = " ".join(map(str, self.k_points))
            f.write(f"    {k_points_str}\n\n")

            f.write('ATOMIC_POSITIONS angstrom\n')
            for atom in self.atoms:
                f.write(
                    f"    {atom[0]:<5} {atom[1]:14.10f} "
                    f"{atom[2]:14.10f} {atom[3]:14.10f}\n"
                )


def process_trajectory(system, traj_file):
    if system not in MATERIAL_CONFIG:
        raise ValueError(f"Material {system} not implemented")

    with open(traj_file, 'r') as f:
        lines = f.readlines()
    nat = int(lines[0])
    nframes = len(lines) // (nat + 2)
    
    for frame in range(nframes):
        # Get frame data directly from lines
        frame_lines = lines[frame * (nat + 2):(frame + 1) * (nat + 2)]
        frame_data = ''.join(frame_lines)
            
        for i, E in enumerate(['0', 'x', 'y', 'z']):
            dir_path = f"{traj_file[:-4]}-E{E}-{frame}"
            outdir = OUTDIR + dir_path
            os.makedirs(dir_path, exist_ok=True)
            
            # Get electric field magnitude from material config
            efield_mag = MATERIAL_CONFIG[system]['efield_magnitude']
            efield = (
                [efield_mag if j == i - 1 else 0.0 for j in range(3)]
                if i > 0 else [0.0] * 3
            )
            
            # Create QEGenerator instance
            output_file = f"{dir_path}/{system}-scf.inp"
            qe = QEGenerator(
                output_file, system, PSEUDO_DIR, outdir, frame_data
            )
            
            # Set control parameters
            qe.set_params(
                qe.control_params,
                lelfield='.true.',
                nberrycyc=1
            )
            
            # Set system parameters
            system_params = {
                k: v for k, v in MATERIAL_CONFIG[system].items()
                if k in ['ibrav', 'nbnd']
            }
            qe.set_params(qe.system_params, **system_params)
            
            # Set electron parameters with electric field
            qe.set_params(
                qe.electron_params,
                efield_cart_1=efield[0],
                efield_cart_2=efield[1],
                efield_cart_3=efield[2]
            )
            
            # Set atomic species and k-points
            qe.set_atomic_species(MATERIAL_CONFIG[system]['species'])
            qe.set_k_points(MATERIAL_CONFIG[system]['k_points'])
            qe.set_atomic_positions()
            qe.write_qe_input()


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculator_QE.py <material>")
        sys.exit(1)
        
    system = sys.argv[1]
    traj_files = {
        'SiO2': ["SiO2/SiO2-T300.xyz", "SiO2/SiO2-T600.xyz"],
        'BaTiO3': ["BaTiO3/BaTiO3.xyz", "BaTiO3/BaTiO3-wall.xyz"]
    }
    
    if system not in traj_files:
        print("Material not implemented")
        sys.exit(1)
        
    for traj in traj_files[system]:
        process_trajectory(system, traj)


if __name__ == '__main__':
    main()
