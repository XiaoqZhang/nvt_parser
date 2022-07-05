import os
import re
from turtle import update
import numpy as np 
import pandas as pd
import math
import click
import json

from pymatgen.core import Structure

import matplotlib.pyplot as plt

# Files required
#   | structure_folder
#   | -- simulation.input
#   | -- Movies/System_0
#   |    | -- Framework_0_final.vasp
#        | -- Movie_%s_%d.%d.%d_298.000000_0.000000_allcomponents.pdb

ch4_sigma , ch4_epsilon = 3.73, 148
uff = pd.read_csv("./ff_data/uff.csv")

nvt_path = "./nvt_results"
rst_path = "./parse_results"

def get_supercell(structure):
    with open(os.path.join(nvt_path, "%s/simulation.input" %structure)) as f_input:
        for line in f_input:
            if re.search("UnitCells", line):
                s1, s2, s3 = line.split()[1], line.split()[2], line.split()[3]
    supercell = [int(s1), int(s2), int(s3)]
    return supercell

def lj(sigma_lb, epsilon_lb, distance):
    u_lj = 4 * epsilon_lb * ((sigma_lb/distance)**12 - (sigma_lb/distance)**6)
    return u_lj

def violin_plot(structure, atom_lst, weights):
    key_weight = {}
    for idx in range(len(weights)):
        if atom_lst[idx] not in key_weight:
            key_weight.update({atom_lst[idx]: weights[idx]})
        else:
            key_weight[atom_lst[idx]] = np.append(key_weight[atom_lst[idx]], weights[idx])
    ele = [key for key in key_weight.keys()]
    data = [key_weight[key] for key in key_weight.keys()]
    pos = np.arange(len(ele))

    fs = 15  # fontsize
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    axs.violinplot(data, pos, widths=0.3,
                     showmeans=True, showextrema=True, showmedians=True)
    axs.plot([0,len(ele)-1], [0, 0], linestyle='dashed')
    axs.set_xticks(np.arange(len(ele)))
    axs.set_xticklabels(ele)
    axs.tick_params(labelsize=fs, labelrotation=45)
    axs.set_ylabel("weight", fontdict={"fontsize": 20})
    fig.savefig(os.path.join(rst_path, "violin_plots/%s.png" %structure))
    plt.close(fig)
    return None

def dist_plot(structure, potential):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    axs.hist(potential, bins=100)
    axs.set_xlabel("Lennard-Jones potential [kJ/mol]", fontdict={"fontsize": 20})
    axs.set_ylabel("Times", fontdict={"fontsize": 20})
    fig.savefig(os.path.join(rst_path, "dist_plots/%s.png" %structure))
    plt.close(fig)
    return None

def scale_pot_plot(structure, po_ori, po_scl):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    axs.scatter(po_ori, po_scl, s=5, marker="x")
    axs.set_xlabel("Lennard-Jones potential [kJ/mol]", fontdict={"fontsize": 20})
    axs.set_ylabel("Scaled Lennard-Jones potential", fontdict={"fontsize": 20})
    fig.savefig(os.path.join(rst_path, "scale_po_plots/%s.png" %structure))
    plt.close(fig)
    return None

def read_cif_atom(structure):
    with open(os.path.join(nvt_path, "%s/%s.cif" %(structure,structure))) as f_framework:
        lines = f_framework.readlines()
        lines = lines[23:]
        atom_cif = [line.split()[0] for line in lines]
        return atom_cif
    
def atom_label(cif_list, sp_list):
    ele = np.unique(np.array(cif_list))
    cif_list_label = cif_list.copy()
    sp_list_label = sp_list.copy()
    count = [0] * len(ele)
    sp_count = [0] * len(ele)
    for idx in range(len(cif_list)):
        for idj in range(len(ele)):
            if cif_list[idx] == ele[idj]:
                count[idj] += 1
                cif_list_label[idx] = cif_list[idx] + str(count[idj])
    for idx in range(len(sp_list)):
        for idj in range(len(ele)):
            if sp_list[idx] == ele[idj]:
                if sp_count[idj] < count[idj]:
                    sp_count[idj] += 1
                else:
                    sp_count[idj] = 1
                sp_list_label[idx] = sp_list[idx] + str(sp_count[idj])       

    return cif_list_label, sp_list_label

#@click.command()
#@click.option("--structure")

# Run the functions
def run(structure):

    sc = get_supercell(structure)
    sp_cry = Structure.from_file(os.path.join(nvt_path, "%s/Movies/System_0/Framework_0_final.vasp" %structure))
    #un_cry = sp_cry.get_primitive_structure()
    un_cry  = sp_cry

    #if (sc == [1, 1, 1]) and (len(sp_cry.sites) != len(un_cry.sites)):
        #un_cry = sp_cry

    site_sigma = [uff.loc[uff["element"] == site.specie.symbol]["lb_sigma"].values for site in un_cry.sites]
    site_epsilon = [uff.loc[uff["element"] == site.specie.symbol]["lb_epsilon"].values for site in un_cry.sites]
    for i in range(len(un_cry.sites)):
        un_cry.sites[i].properties = {"lb_sigma": site_sigma[i], "lb_epsilon": site_epsilon[i], "weight": []}
    
    with open(os.path.join(nvt_path, "%s/Movies/System_0/Movie_%s_%d.%d.%d_298.000000_0.000000_allcomponents.pdb" %(structure, structure, sc[0], sc[1], sc[2]))) as file:
        data = file.readlines()
        atom_pos = np.array([np.array([float(i) for i in line.split()[4:7]]) for line in data if "ATOM" in line])

    threshold = 7
    sphere_sites = np.array([[site for site in un_cry.get_sites_in_sphere(pt=pos, r=threshold) if site.specie.symbol!="H"] 
                                for pos in atom_pos], dtype=object)
    #sphere_sites = np.array([un_cry.get_sites_in_sphere(pt=pos, r=threshold) for pos in atom_pos], dtype=object)

    dist = np.array([np.array([site.distance_from_point(atom_pos[idx]) for site in sphere_sites[idx]]) 
                        for idx in range(len(atom_pos))], dtype=object)
    potential = np.array([np.array([lj(sphere_sites[i][j].properties["lb_sigma"], sphere_sites[i][j].properties["lb_epsilon"], dist[i][j]) 
                            if sphere_sites[i][j].specie.symbol!="H" else np.array([0])
                            for j in range(len(sphere_sites[i]))]).flatten() 
                            for i in range(len(atom_pos))], dtype=object)
    po_ana = np.hstack(potential)

    min = po_ana.min()
    max = po_ana.max()
    po_norm = -((po_ana - min) / (max - min) - 1)
    #po_sigmoid = np.exp(-po_ana/abs(np.min(po_ana))) / np.sum(np.exp(-po_ana/abs(np.min(po_ana))))

    dist_plot(structure, po_ana)
    scale_pot_plot(structure, po_ana, po_norm)

    idx = 0
    for i in range(sphere_sites.shape[0]):
        for j in range(len(sphere_sites[i])):
            sphere_sites[i][j].properties["weight"].append(po_norm[idx])
            idx += 1
    weights = np.array([np.sum(site.properties["weight"]) for site in un_cry.sites])
    atom_lst = [site.specie.symbol for site in un_cry.sites]

    atom_cif = read_cif_atom(structure)
    atom_cif_label, atom_lst_label = atom_label(atom_cif, atom_lst)

    cif_weight = {}
    for i in range(len(atom_cif_label)):
        for j in range(len(atom_lst_label)):
            if atom_cif_label[i] == atom_lst_label[j]:
                if atom_cif_label[i] not in cif_weight:
                    cif_weight.update({atom_cif_label[i]: weights[j]})
                else:
                    cif_weight[atom_cif_label[i]] += weights[j]
    
    uc_weight = [cif_weight[i] for i in cif_weight]
    uc_weight = [w/np.array(uc_weight).max() for w in uc_weight]
    cif_weight = {}
    for i, j in zip(atom_cif_label, uc_weight):
        cif_weight.update({i: j})
    violin_plot(structure, atom_cif, uc_weight)

    with open(os.path.join(rst_path, "weight_json/%s.json" %structure), 'w') as json_file:
        json.dump(cif_weight, json_file)

if __name__ == '__main__':
    structure_lst = os.listdir(nvt_path)
    for structure in structure_lst:
        run(structure)
