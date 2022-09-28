from cmath import nan
from copy import copy
import os
import re
from turtle import distance, update
import numpy as np 
import pandas as pd
import math
import click
import json
import concurrent.futures

from pymatgen.core import Structure

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# Files required
#   | structure_folder
#   | -- simulation.input
#   | -- structure.cif
#   | -- Movies/System_0
#   |    | -- Framework_0_final.vasp
#        | -- Movie_%s_%d.%d.%d_298.000000_0.000000_allcomponents.pdb

ch4_sigma , ch4_epsilon = 3.73, 148
c_sigma, c_epsilon = 2.8, 27.0
o_sigma, o_epsilon = 3.05, 79.0
uff = pd.read_csv("./ff_data/uff.csv")

threshold = 20

#nvt_path = "/run/user/1001/gvfs/smb-share:server=lsmosrv2.epfl.ch,share=xiazhang/core_ch4_nvt"
#nvt_path = "./nvt_results"
nvt_path = "/home/xiaoqi/molsim/co2_nvt_deduplicated"
rst_path = "./parse_results_co2"

def min_max(numbers):
    a = np.min(numbers)
    b = np.max(numbers)
    return (numbers-a)/(b-a)

def get_supercell(structure):
    with open(os.path.join(nvt_path, "%s/simulation.input" %structure), "r") as f_input:
        for line in f_input:
            if re.search("UnitCells", line):
                s1, s2, s3 = line.split()[1], line.split()[2], line.split()[3]
    supercell = [int(s1), int(s2), int(s3)]
    return supercell

def get_lattice(structure):
    with open(os.path.join(nvt_path, "%s/Movies/System_0/Framework_final.pdb" %structure)) as file:
        lines = file.readlines()
        lattice = np.array([float(l) for l in lines[1].split()[1:4]])
        frame_ele = np.array([l.split()[2] for l in lines[2:]])
        frame_pos = np.array([[float(t) for t in l.split()[4:7]] for l in lines[2:]])
    return lattice, frame_ele, frame_pos

def plot_distance_distribution(structure, dist):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    axs.hist(dist, bins=100)
    axs.set_xlabel("Distance (A)", fontdict={"fontsize": 20})
    axs.set_ylabel("Times", fontdict={"fontsize": 20})
    fig.savefig(os.path.join(rst_path, "dist_plots/%s.png" %structure))
    plt.close(fig)
    return None

def plot_lj_distribution(structure, poten):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    axs.hist(poten, bins=100)
    axs.set_xlabel("Normalized Lennard-Jones potential", fontdict={"fontsize": 20})
    axs.set_ylabel("Times", fontdict={"fontsize": 20})
    fig.savefig(os.path.join(rst_path, "poten_plots/%s.png" %structure))
    plt.close(fig)
    return None

    
# Run the functions
def run(structure):
    sc = get_supercell(structure)
    latt, ele, pos = get_lattice(structure)

    # read cif structure
    cif = Structure.from_file(os.path.join(nvt_path, "%s/%s.cif" %(structure, structure)))

    # read gas position
    with open(os.path.join(nvt_path, "%s/Movies/System_0/Movie_%s_%d.%d.%d_298.000000_0.000000_allcomponents.pdb" %(structure, structure, sc[0], sc[1], sc[2]))) as file:
        data = file.readlines()
        positions = np.array([line.split() for line in data if "ATOM" in line])
        pos_c = np.array([[float(t) for t in l[4:7]] for l in positions if l[1] == '2'])
        pos_o1 = np.array([[float(t) for t in l[4:7]] for l in positions if l[1] == '1'])
        pos_o2 = np.array([[float(t) for t in l[4:7]] for l in positions if l[1] == '3'])
    
    # get sites withine the cutoff range
    nn = np.array([cif.get_sites_in_sphere(c, threshold) for c in pos_c], dtype=object)
    nn_dist = np.array([[n.nn_distance for n in s] for s in nn], dtype=object)

    # get uff parameters
    sigma_c = np.array([(uff[uff["element"]==s.specie.symbol]["sigma"].item()+c_sigma)/2 for s in cif.sites])
    epsilon_c = np.array([np.sqrt(uff[uff["element"]==s.specie.symbol]["epsilon"].item()*c_epsilon) for s in cif.sites])

    sigma_o = np.array([(uff[uff["element"]==s.specie.symbol]["sigma"].item()+o_sigma)/2 for s in cif.sites])
    epsilon_o = np.array([np.sqrt(uff[uff["element"]==s.specie.symbol]["epsilon"].item()*o_epsilon) for s in cif.sites])

    for i in range(len(cif.sites)):
        cif.sites[i].properties = {
            "sigma_c": sigma_c[i], "epsilon_c": epsilon_c[i], 
            "sigma_o": sigma_o[i], "epsilon_o": epsilon_o[i]
            "weight": []
        }
    '''
    # calculate distance with periodic boundary conditions
    dist_c = np.array([np.abs([atom-c for atom in pos]) for c in pos_c])
    dist_c = np.array([[[l-r if r>l/2 else r for r,l in zip(a, latt)] for a in s] for s in dist_c])
    dist_c *= dist_c
    dist_c = np.sum(dist_c, axis=2)
    dist_c = np.sqrt(dist_c)

    dist_o1 = np.array([np.abs([atom-o for atom in pos]) for o in pos_o1])
    dist_o1 = np.array([[[l-r if r>l/2 else r for r,l in zip(a, latt)] for a in s] for s in dist_o1])
    dist_o1 *= dist_o1
    dist_o1 = np.sum(dist_o1, axis=2)
    dist_o1 = np.sqrt(dist_o1)
    
    dist_o2 = np.array([np.abs([atom-o for atom in pos]) for o in pos_o2])
    dist_o2 = np.array([[[l-r if r>l/2 else r for r,l in zip(a, latt)] for a in s] for s in dist_o2])
    dist_o2 *= dist_o2
    dist_o2 = np.sum(dist_o2, axis=2)
    dist_o2 = np.sqrt(dist_o2)
    '''

    # read sigma and epsilon
    ele_sigma_c = np.array([(uff[uff["element"]==e]["sigma"].item()+c_sigma)/2 for e in ele])
    ele_epsilon_c = np.array([np.sqrt(uff[uff["element"]==e]["epsilon"].item()*c_epsilon) for e in ele])

    ele_sigma_o = np.array([(uff[uff["element"]==e]["sigma"].item()+o_sigma)/2 for e in ele])
    ele_epsilon_o = np.array([np.sqrt(uff[uff["element"]==e]["epsilon"].item()*o_epsilon) for e in ele])

    # calculate lj potentials
    poten_c = np.array([ele_epsilon_c*((ele_sigma_c/r)**12-(ele_sigma_c/r)**6) for r in dist_c])
    poten_o1 = np.array([ele_epsilon_o*((ele_sigma_o/r)**6-(ele_sigma_o/r)**12) for r in dist_o1])
    poten_o2 = np.array([ele_epsilon_o*((ele_sigma_o/r)**6-(ele_sigma_o/r)**12) for r in dist_o2])

    avg_poten_c = np.average(poten_c, axis=0)
    avg_poten_o1 = np.average(poten_o1, axis=0)
    avg_poten_o2 = np.average(poten_o2, axis=0)

    avg_poten = np.array([c+o1+o2 for c, o1, o2 in zip(avg_poten_c, avg_poten_o1, avg_poten_o2)])

    # map the calculated potentials to one unit cell
    num_uc = sc[0] * sc[1] * sc[2]
    num_at = int(len(pos) / num_uc)
    for i in range(num_uc):
        if i == 0:
            dist_uc = dist_c[0:num_at]
            poten_uc = avg_poten[0:num_at]
        else:
            dist_uc += dist_c[i*num_at:(i+1)*num_at]
            poten_uc += avg_poten[i*num_at:(i+1)*num_at]
    dist_uc /= num_uc
    poten_uc /= num_uc

    # normalization
    scaler_poten = MinMaxScaler()
    poten_norm = scaler_poten.fit_transform(poten_uc.reshape(-1, 1))

    scaler_dist = MinMaxScaler()
    dist_norm = scaler_dist.fit_transform(dist_uc.reshape(-1, 1))

    plot_distance_distribution(structure, dist_norm)
    plot_lj_distribution(structure, poten_norm)

    return {
        structure: {
            "dist": list(dist_norm.reshape(-1,)),
            "poten": list(poten_norm.reshape(-1,))
        }
    }


if __name__ == '__main__':
    structure_lst = os.listdir(nvt_path)
    dumped = {}
    for structure in structure_lst:
        if os.path.exists(os.path.join(nvt_path, "%s/Movies/System_0/Framework_0_final.vasp" %structure)) == False:
            print("%s undone!" %structure)
            structure_lst.remove(structure)
       
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        result = list(executor.map(run, structure_lst))
    for r in result:
        dumped.update(r)
    with open(os.path.join(rst_path, "weights.json"), "w") as file:
        json.dump(dumped, file, indent=4)
