import os
import sys
os.chdir('/home/csy/work/3D/PharDiff')
sys.path.append("/home/csy/work/3D/PharDiff/utils")
sys.path.append("/home/csy/work/3D/PharDiff")
import argparse
import multiprocessing as mp
import pickle
import shutil
from functools import partial
from tqdm.auto import tqdm
from utils.data import PDBProtein, parse_sdf_file
from mol_tree import *
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from openbabel import pybel
from openbabel import openbabel as ob
import subprocess
import os
import numpy as np
import shutil
from plip.structure.preparation import PDBComplex
import traceback
import utils.misc as misc
import utils.transforms as trans
from torch_geometric.transforms import Compose
from models.diffusion import PharDiff, log_sample_categorical
from scripts.sample_diffusion import sample_diffusion_ligand
import argparse
import os, sys
sys.path.append("/home/csy/work/3D/PharDiff")
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
from datasets.pl_data import ProteinLigandData, torchify_dict
import argparse
import os, sys
sys.path.append("/home/csy/work/3D/PharDiff")
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask

eval_step = -1
#r = torch.load('/home/csy/work/3D/PharDiff/scripts/pdgfr/1h00_B_FAP_result.pt')

#result_path = '/home/csy/work/3D/PharDiff/scripts/PDGFRb/data/ligand_result_PINN_1000.pickle'
#result_path = '/home/csy/work/3D/PharDiff/scripts/PDGFRb/data/ligand_result_PINN_1000.pickle'
result_path = '/home/csy/work/3D/PharDiff/scripts/pdgfr/1h00_B_FAP_result.pickle'
with open(result_path, 'rb') as fr:
    r = pickle.load(fr)

all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
all_pred_ligand_v = r['pred_ligand_v_traj']
num_samples = len(all_pred_ligand_pos) 
num_samples = 0
all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
n_recon_success, n_eval_success, n_complete = 0, 0, 0
results = []
all_pair_dist, all_bond_dist = [], []
all_atom_types = Counter()
success_pair_dist, success_atom_types = [], Counter()
cnt = 0
Total = 0
invalid = 0
valid = 0
verbose=True
atom_enc_mode = 'add_aromatic'
docking_mode = 'vina_score'
#     for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
for sample_idx, (pred_pos, pred_v) in enumerate(tqdm(zip(all_pred_ligand_pos, all_pred_ligand_v), desc='docking')):
    pred_pos, pred_v = pred_pos[eval_step], pred_v[eval_step]
    #print(pred_pos)
    # stability check
    pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)

    all_atom_types += Counter(pred_atom_type)

    r_stable = analyze.check_stability(pred_pos, pred_atom_type)
    all_mol_stable += r_stable[0]
    all_atom_stable += r_stable[1]
    all_n_atom += r_stable[2]

    pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
    all_pair_dist += pair_dist
    try:
        pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
        mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        smiles = Chem.MolToSmiles(mol)
    except:
        continue
    n_recon_success += 1
    if '.' in smiles:
        #print("Separation error.")
        continue
    try:
        Chem.SanitizeMol(mol)
        valid += 1 
    except:
        #print("Valid error.")
        invalid += 1
        continue            
    n_complete += 1
    try:
        chem_results = scoring_func.get_chem(mol)
        Total += 1
    except:
        cnt += 1
        Total += 1
        print("Valence error(Count of Explicit valence for atom) : ", cnt,"/",Total)
        continue
    #try:
        #chem_results = scoring_func.get_chem(mol)
    if docking_mode == 'qvina':
        vina_task = QVinaDockingTask.from_generated_mol(
            mol, r['data'].ligand_filename, protein_root=args.protein_root)
        vina_results = vina_task.run_sync()
    elif docking_mode in ['vina_score', 'vina_dock']:
        vina_task = VinaDockingTask.from_generated_mol(
            mol, r['data'].ligand_filename, 
            protein_root ='/home/csy/work/3D/PharDiff/scripts/pdgfr/1h00_pocket.pdb') # , protein_root=args.protein_root
        score_only_results = vina_task.run(mode='score_only', exhaustiveness=16)
        minimize_results = vina_task.run(mode='minimize', exhaustiveness=16)
        vina_results = {
            'score_only': score_only_results,
            'minimize': minimize_results
        }
        if docking_mode == 'vina_dock':
            docking_results = vina_task.run(mode='dock', exhaustiveness=16)
            vina_results['dock'] = docking_results
    else:
        vina_results = None

    n_eval_success += 1
    # except:
    #     print("docking error")
    #     continue
    
    bond_dist = eval_bond_length.bond_distance_from_mol(mol)
    success_pair_dist += pair_dist
    success_atom_types += Counter(pred_atom_type)
    
    results.append({
        'mol': mol,
        'smiles': smiles,
        'ligand_filename': r['data'].ligand_filename,
        'pred_pos': pred_pos,
        'pred_v': pred_v,
        'chem_results': chem_results,
        'vina': vina_results
    })
    
qed = [r['chem_results']['qed'] for r in results]
sa = [r['chem_results']['sa'] for r in results]
logp = [r['chem_results']['logp'] for r in results]
li = [r['chem_results']['lipinski'] for r in results]
print('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
print('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
print('LogP:    Mean: %.3f Median: %.3f' % (np.mean(logp), np.median(logp)))
print('Lipinski:    Mean: %.3f Median: %.3f' % (np.mean(li), np.median(li)))
print('VALID:    Mean: %.3f' % (valid/(invalid+valid)))

if docking_mode in ['vina_dock', 'vina_score']:
    vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
    vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
    print('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
    print('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
    
    print('Vina Score:  Min: %.3f ' % (np.min(vina_score_only)))
    print('Vina Min  :  Min: %.3f ' % (np.min(vina_min)))
    
    if docking_mode == 'vina_dock':
        vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
        print('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))
        print('Vina Dock :  Min: %.3f ' % (np.min(vina_dock)))

if docking_mode == 'vina_dock':
    ind = np.argmin(vina_dock)
else:
    ind = np.argmin(vina_score_only)
print('QED:   [Docking Top1]: %.3f ' % (qed[ind]))
print('SA:    [Docking Top1]: %.3f ' % (sa[ind]))
print('LogP:    [Docking Top1]: %.3f ' % (logp[ind]))
print('Lipinski:    [Docking Top1]: %.3f ' % (li[ind]))

# import pickle
# with open('/home/csy/work/3D/PharDiff/scripts/PDGFRb/results/result_PINN_ref_100.pickle', 'wb') as f:
#     pickle.dump(results, f)