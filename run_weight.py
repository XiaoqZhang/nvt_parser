from cmath import nan
from copy import copy
import os
import re
from turtle import update
import numpy as np 
import pandas as pd
import math
import click
import json
import concurrent.futures

from pymatgen.core import Structure

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
nvt_path = "./nvt_results"
rst_path = "./parse_results_co2"

def get_supercell(structure):
    with open(os.path.join(nvt_path, "%s/simulation.input" %structure)) as f_input:
        for line in f_input:
            if re.search("UnitCells", line):
                s1, s2, s3 = line.split()[1], line.split()[2], line.split()[3]
    supercell = [int(s1), int(s2), int(s3)]
    return supercell

def get_lattice(structure):
    with open(os.path.join(nvt_path, "%s/Movies/System_0/Framework_0_final.vasp" %structure)) as file:
        lines = file.readlines()
        lines = lines[2:5]
        lattice = [[float(l) for l in line.split()] for line in lines]
    return lattice

def lj(sigma_lb, epsilon_lb, distance):
    u_lj = 4 * epsilon_lb * ((sigma_lb/distance)**12 - (sigma_lb/distance)**6)
    return u_lj

# Run the functions
def run(structure):
    print("Pharsing %s" %structure)
    sc = get_supercell(structure)
    un_cry = Structure.from_file(os.path.join(nvt_path, "%s/%s.cif" %(structure, structure)))

    latt = get_lattice(structure)
    un_cry.lattice = [np.array(latt[0])/sc[0], np.array(latt[1])/sc[1], np.array(latt[2])/sc[2]]


    ele_sigma = [uff.loc[uff["element"] == site.specie.symbol]["sigma"].values for site in un_cry.sites]
    ele_epsilon = [uff.loc[uff["element"] == site.specie.symbol]["epsilon"].values for site in un_cry.sites]
    site_sigma_c = [(s+c_sigma)/2 for s in ele_sigma]
    site_epsilon_c = [math.sqrt(e*c_epsilon) for e in ele_epsilon]
    site_sigma_o = [(s+o_sigma)/2 for s in ele_sigma]
    site_epsilon_o = [math.sqrt(e*o_epsilon) for e in ele_epsilon]
    for i in range(len(un_cry.sites)):
        un_cry.sites[i].properties = {
            "lb_sigma_c": site_sigma_c[i], "lb_epsilon_c": site_epsilon_c[i], 
            "lb_sigma_o": site_sigma_o[i], "lb_epsilon_o": site_epsilon_o[i],
            "weight": []
        }
    
    with open(os.path.join(nvt_path, "%s/Movies/System_0/Movie_%s_%d.%d.%d_298.000000_0.000000_allcomponents.pdb" %(structure, structure, sc[0], sc[1], sc[2]))) as file:
        data = file.readlines()
        lines = [line.split() for line in data if "ATOM" in line]
        atom_pos_c = [np.array([float(t) for t in l[4:7]]) for l in lines if l[1] == '2']
        atom_pos_o1 = [np.array([float(t) for t in l[4:7]]) for l in lines if l[1] == '1']
        atom_pos_o2 = [np.array([float(t) for t in l[4:7]]) for l in lines if l[1] == '3']
    
    sphere_sites = np.array([[site for site in un_cry.get_sites_in_sphere(pt=pos, r=threshold) if site.specie.symbol != "H"] 
                                for pos in atom_pos_c], dtype=object)
    print("atoms in the cutoff range: %s" %sphere_sites)

    dist_c = np.array([np.array([site.distance_from_point(atom_pos_c[idx]) for site in sphere_sites[idx]]) 
                        for idx in range(len(atom_pos_c))], dtype=object)
    dist_o1 = np.array([np.array([site.distance_from_point(atom_pos_o1[idx]) for site in sphere_sites[idx]]) 
                        for idx in range(len(atom_pos_o1))], dtype=object)
    dist_o2 = np.array([np.array([site.distance_from_point(atom_pos_o2[idx]) for site in sphere_sites[idx]]) 
                        for idx in range(len(atom_pos_o2))], dtype=object)
    potential_c = np.array([np.array([lj(sphere_sites[i][j].properties["lb_sigma_c"], sphere_sites[i][j].properties["lb_epsilon_c"], dist_c[i][j]) 
                            for j in range(len(sphere_sites[i]))]).flatten() 
                            for i in range(len(atom_pos_c))], dtype=object)
    potential_o1 = np.array([np.array([lj(sphere_sites[i][j].properties["lb_sigma_o"], sphere_sites[i][j].properties["lb_epsilon_o"], dist_o1[i][j]) 
                            for j in range(len(sphere_sites[i]))]).flatten() 
                            for i in range(len(atom_pos_o1))], dtype=object)
    potential_o2 = np.array([np.array([lj(sphere_sites[i][j].properties["lb_sigma_o"], sphere_sites[i][j].properties["lb_epsilon_o"], dist_o2[i][j]) 
                            for j in range(len(sphere_sites[i]))]).flatten() 
                            for i in range(len(atom_pos_o2))], dtype=object)
    potential = np.sum([potential_c, potential_o1, potential_o2], axis=0)
    print(potential)

    np.seterr(invalid='ignore')
    mins = [po.min() if len(po)!=0 else np.nan for po in potential]
    maxs = [po.max() if len(po)!=0 else np.nan for po in potential]
    po_norm = np.array([-((po-min)/(max-min)-1) for min, max, po in zip(mins, maxs, potential)], dtype=object)
    po_norm = np.hstack(po_norm)

    idx = 0
    for i in range(sphere_sites.shape[0]):
        for j in range(len(sphere_sites[i])):
            sphere_sites[i][j].properties["weight"].append(po_norm[idx])
            idx += 1
    weights = np.array([np.average(site.properties["weight"]) if len(site.properties["weight"])!=0 else 0 
                            for site in un_cry.sites])
    np.nan_to_num(weights, copy=False, nan=0)
    
    print("%s Done" %structure)
    return {structure: list(weights)}


if __name__ == '__main__':
    structure_lst = os.listdir(nvt_path)
    dumped = {}
    for structure in structure_lst:
        if os.path.exists(os.path.join(nvt_path, "%s/Movies/System_0/Framework_0_final.vasp" %structure)) == False:
            print("%s undone!" %structure)
            structure_lst.remove(structure)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        result = list(executor.map(run, structure_lst))
    for r in result:
        dumped.update(r)
    with open(os.path.join(rst_path, "weights_%s.json" %threshold), "w") as file:
        json.dump(dumped, file, indent=4)
