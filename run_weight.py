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
uff = pd.read_csv("./ff_data/uff.csv")

threshold = 12

nvt_path = "/run/user/1001/gvfs/smb-share:server=lsmosrv2.epfl.ch,share=xiazhang/core_ch4_nvt"
rst_path = "./parse_results"

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


#@click.command()
#@click.option("--structure")

# Run the functions
def run(structure):
    sc = get_supercell(structure)
    un_cry = Structure.from_file(os.path.join(nvt_path, "%s/%s.cif" %(structure, structure)))

    latt = get_lattice(structure)
    un_cry.lattice = [np.array(latt[0])/sc[0], np.array(latt[1])/sc[1], np.array(latt[2])/sc[2]]

    site_sigma = [uff.loc[uff["element"] == site.specie.symbol]["lb_sigma"].values for site in un_cry.sites]
    site_epsilon = [uff.loc[uff["element"] == site.specie.symbol]["lb_epsilon"].values for site in un_cry.sites]
    for i in range(len(un_cry.sites)):
        un_cry.sites[i].properties = {"lb_sigma": site_sigma[i], "lb_epsilon": site_epsilon[i], "weight": []}
    
    with open(os.path.join(nvt_path, "%s/Movies/System_0/Movie_%s_%d.%d.%d_298.000000_0.000000_allcomponents.pdb" %(structure, structure, sc[0], sc[1], sc[2]))) as file:
        data = file.readlines()
        atom_pos = np.array([np.array([float(i) for i in line.split()[4:7]]) for line in data if "ATOM" in line])

    sphere_sites = np.array([[site for site in un_cry.get_sites_in_sphere(pt=pos, r=threshold) if site.specie.symbol != "H"] 
                                for pos in atom_pos], dtype=object)

    dist = np.array([np.array([site.distance_from_point(atom_pos[idx]) for site in sphere_sites[idx]]) 
                        for idx in range(len(atom_pos))], dtype=object)
    potential = np.array([np.array([lj(sphere_sites[i][j].properties["lb_sigma"], sphere_sites[i][j].properties["lb_epsilon"], dist[i][j]) 
                            for j in range(len(sphere_sites[i]))]).flatten() 
                            for i in range(len(atom_pos))], dtype=object)

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
    with open(os.path.join(rst_path, "weights.json"), "w") as file:
        json.dump(dumped, file, indent=4)
