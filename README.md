# LAMMPS_Extensions
This is a collection of the extensions (fixes, computes, potentials) I've created for the MD simulation software package [LAMMPS](https://github.com/lammps/) [1,2].

## How to install

Simply copy the matching header (.h) and implementation files (.cpp) that you wish to use into the `src/` folder of your LAMMPS installation and compile LAMMPS using the make option of your choice

You can then invoke them in your LAMMPS scripts according to the usual naming conventions.
The header files list the name inside the LAMMPS scripts as the first parameter of `PairStyle(...)`, `ComputeStyle(...)`, or `FixStyle(...)`, respectively.

## Class II Soft Potentials

- `pair_style lj/class2/nosoft/coul/cut/soft`
- `pair_style lj/class2/soft/coul/cut/nosoft`
- `pair_style lj/class2/soft/`
- `pair_style lj/class2/soft/deriv`
- `pair_style coul/cut/soft/deriv`

This contains a series of soft 9–6 Lennard–Jones and Coulomb pair potentials [3] that are compatible with Class II force fields (PCFF, COMPASS, ...) and which can be used to perform thermodynamic integration (TI) with less instabilities and improved sampling of the full phase space.

I've originally adapted these from the soft potentials of the [USER-FEP](https://github.com/agiliopadua/compute_fep) package (developed by [Agilio Padua](https://github.com/agiliopadua/)) but now for Class II force fields.

By now, LAMMPS has added such Class II Soft Potentials to the official distribution, indepently from the code here.

They use a similar or sometimes the same notation as here.

Here are also potentials provided, which output the derivatives with respect to the parameter <img src="https://render.githubusercontent.com/render/math?math=\lambda"> (the coupling parameter of TI), which should only be used during a `rerun` of a trajectory in LAMMPS.

## Computes

- `compute asphericity`
- `compute asphericity/chunk`
- `compute ellipsoid/chunk`
- `compute nematic/chunk`

These computes calculate properties related to orientation, shape, and structure of atom groups (or chunk) in an MD simulation.

They can be used to determine the asphericity of a molecule, the principal axes and their directions, and the nematic order parameter of a collection of molecules.

In the current form, this code is mostly intended for atomistic simulations (`atom_style full`) and is not tested to work with aspherical particles (Gay–Berne particles, etc.).

## Fixes

- `fix chicken/chunk`

Here, `fix chicken/chunk` is given, which (despite the silly name) is developed for atomistic simulations of azobenzene-containing materials under UV-Vis irradiation.

This fix exerts an angle-dependent torque arising from an effective orientation potential that is acting on an azo-chromophore under irradiation. This potential is implemented based on the so-called orientation approach [4].

## To do

- Write documentation for the code
- Provide examples
- Clean up the code and submit it to the LAMMPS project



## References

(Part of) this code has been used in the following publications:

- M. Koch, M. Saphiannikova, S. Santer, O. Guskova, [*Phys. Chem. B.* (2017), 121 (37), 8854–8867.](https://pubs.acs.org/doi/10.1021/acs.jpcb.7b07350)
- M. Koch, M. Saphiannikova, O. Guskova, [*Langmuir* (2019), 35 (45), 14659–14669.](https://pubs.acs.org/doi/abs/10.1021/acs.langmuir.9b02960)


### Sources 

- [1] S. Plimpton, [*J. Comput. Phys.* (1995), 117, 1–19.](https://doi.org/10.1006/jcph.1995.1039)
- [2] https://lammps.sandia.gov/
- [3] T. C. Beutler et al., [*Chem. Phys. Lett.* (1994), 222 (6), 529–539.](https://doi.org/10.1016/0009-2614(94)00397-1)
- [4] V. Toshchevikov, J. Ilnytskyi, M. Saphiannikova, [*J. Phys. Chem. Lett.* (2017), 8, 1094–1098.](https://pubs.acs.org/doi/10.1021/acs.jpclett.7b00173)
