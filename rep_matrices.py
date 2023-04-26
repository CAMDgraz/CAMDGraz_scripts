#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel Platero Rochart [daniel.platero-rochart@medunigraz.at]
"""

import pandas as pd
import numpy as np
from ase.io import read
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP
import sys


def read_input(file_name, process):
    """
    Read a file with the paths to the input data

    Parameters:
        file_name: str
            File containing the absolute or relative paths to the input
            structures
        n_rows: int
            Number of rows to read from the file

    Returns:
        depend_var: np.array
            Array containing the values of the dependent variable
        struct_paths: np.array
            Array containing the paths to the input chemical structures
    """
    if process == 'train':
        inp_data = pd.read_csv(file_name, delim_whitespace=True, header=None)
        struct_paths = np.asarray(inp_data.iloc[:, 0], dtype=str)
        depend_var = np.asarray(inp_data.iloc[:, 1],
                                dtype=float).reshape((len(inp_data.iloc[:, 1]),
                                                      1))
        return depend_var, struct_paths

    if process == 'predict':
        inp_data = pd.read_csv(file_name, delim_whitespace=True, header=None)
        struct_paths = np.asarray(inp_data.iloc[:, 0], dtype=str)
        return struct_paths


def coulomb_matrix(structures, n_atoms):
    """
    Function to obtain the Coulomb matrix of a system using Dscribe and ASE.
    By default no permutation is done

    Parameters:
        structures: iterable
            Iterable containing the paths to the chemical structures.
        n_atoms: int
            Number of atoms in the chemical structures.

    Returns:
        cm_matrices: np.array
            2D array where each row correspond to the CM matrix (1D vector) of
            a given chemical structure.
    """
    cm = CoulombMatrix(n_atoms_max=n_atoms, permutation='none')
    cm_matrices = np.zeros((len(structures), n_atoms*n_atoms), dtype=float)
    for idx, inp in enumerate(structures):
        mol = read(inp)
        feature = cm.create(mol, n_jobs=2, verbose=True)
        cm_matrices[idx] = feature.ravel()

    return cm_matrices


def acsf_matrix(structures):
    """
    Function to obtain the ACSF matrix of a system using Dscribe and ASE.

    Parameters:
        structures: iterable
            Iterable containing the paths to the chemical structures.

    Returns:
        acsf_matrices: np.array
            2D array where each row correspond to the ACSF matrix.
    """
    structure = read(structures[0])
    species = set(structure.get_chemical_symbols())
    acsf = ACSF(species=species, rcut=5, sparse=False)
    acsf_matrices = []
    for inp in structures:
        mol = read(inp)
        feature = acsf.create(mol, n_jobs=1)
        acsf_matrices.append(feature.ravel())

    return acsf_matrices


def soap_matix(structures):
    """
    Function to obtain the SOAP matrix of a system using Dscribe and ASE.

    Parameters:
        structures: iterable
            Iterable containing the paths to the chemical structures.

    Returns:
        soap_matrices: np.array
            2D array where each row correspond to the SOAP matrix.
    """
    structure = read(structures[0])
    species = set(structure.get_chemical_symbols())
    soap = SOAP(species=species, nmax=15, lmax=8, rcut=13, sparse=False)
    soap_matrices = []
    for inp in structures:
        mol = read(inp)
        feature = soap.create(mol, n_jobs=1)
        soap_matrices.append(feature.ravel())

    return soap_matrices


if __name__ == '__main__':
    input_file = str(sys.argv[1])  # File with structures path and y variable
    n_atoms = int(sys.argv[2])     # Number of atoms in the molecule
    out_matrix = str(sys.argv[3])  # Name of the output matrix
    representation = str(sys.argv[4])  # name of the representation
    process = str(sys.argv[5])     # train or predict

    if process == 'train':
        energies, input_struct = read_input(input_file, 'train')
        if representation == 'cm':
            matrix = coulomb_matrix(input_struct, n_atoms)
            matrix = np.insert(matrix, [0], energies, axis=1)
        if representation == 'acsf':
            matrix = acsf_matrix(input_struct)
            matrix = np.insert(matrix, [0], energies, axis=1)
        if representation == 'soap':
            matrix = soap_matix(input_struct)
            matrix = np.insert(matrix, [0], energies, axis=1)

    if process == 'predict':
        input_struct = read_input(input_file, 'predict')
        if representation == 'cm':
            matrix = coulomb_matrix(input_struct, n_atoms)
        if representation == 'acsf':
            matrix = acsf_matrix(input_struct)
        if representation == 'soap':
            matrix = soap_matix(input_struct)

    np.savetxt(out_matrix, matrix, delimiter=',')
