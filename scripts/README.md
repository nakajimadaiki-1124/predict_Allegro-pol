# Tutorial for using Allegro-pol to predict vibrational, dielectric, and ferroelectric properties of materials

In this tutorial, we provide a step-by-step guide for pre-processing and post-processing tasks to determine:

- Vibrational and dielectric responses of SiO₂.
- Ferroelectric response of BaTiO₃.

### Pre-processing scripts:

- `calculator_QE.py`: Generates QE input files for calculations under zero field and finite electric fields along x, y, and z. Folder nomenclature is interfaced with `parse_QE.py`.
- `parse_QE.py`: Parses QE XML files to create extxyz file for Allegro-pol, containing energy, forces, polarization, Born charges, and polarizability.

### Post-processing scripts:

- `parity_plot.py`: Plots parity comparisons between DFT data and corresponding Allegro predictions obtained through `nequip-evaluate`.
- `dielectric_constant.py`: Calculates the dielectric constant during a QuantumEspresso or LAMMPS structural relaxation in the presence of an electric field.
- `vibr_dielec_response.py`: Calculates the IR spectrum and the real and imaginary parts of the dielectric constant from MLMD at fixed volume and temperature, performed with the Allegro-pol model.
- `hysteresis.py`: Calculates ferroelectric hysteresis from MLMD at fixed volume and temperature under a sinusoidal electric field, performed with the Allegro-pol model. When provided with multiple datafiles for statistics, the script averages polarization values across all trajectories.
- `dipoles.py`: Parses an extxyz structural file containing Born charges and plots ferroelectric dipoles (2D, 3D plots), angle distributions with respect to the z-axis, and extxyz files with dipoles per unit cell. Works only for orthorhombic perovskite structures.
- `coercive_efield.py`: Plots the coercive field versus temperature, performing statistics at each temperature.
