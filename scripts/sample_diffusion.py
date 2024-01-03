import argparse
import os, sys
import shutil
import time
sys.path.append("./3D-MOL-GENERATION/anonymous")
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.diffusion import PharDiff, log_sample_categorical
from utils.evaluation import atom_num
import random
import pickle 
def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v

with open("./3D-MOL-GENERATION/anonymous/datasets/vocab_super.pickle", "rb") as f:
    voc = pickle.load(f)
prep_vocab = {}
threshold = 0
for key, value in voc.items():    
    if value >= threshold:
        prep_vocab[key] = value
def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', frag='Frag'):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            
            fragnode = []
            fix_node_batch = []
            batch_size = batch.num_graphs
            # for i in range(batch_size):
            #     fix_node = torch.unique(torch.tensor(sum([inter['Fragment_node'] for inter in batch.IntramolInteraction[i]], [])))
            #     #fix_node = torch.unique(torch.tensor(sum([inter['Fragment_node'] for inter in batch.IntramolInteraction[i] if inter['Fragment_smi'] in list(prep_vocab.keys())], [])))
            #     fragnode.append(fix_node)
            #     b = torch.zeros_like(fix_node) + i
            #     fix_node_batch.append(b)
            for i in range(batch_size):
                tar = [inter['Fragment_node'] for inter in batch.IntramolInteraction[i]]
                tar_list = []
                for item in tar:
                    if item not in tar_list:
                        tar_list.append(item)
                if len(tar_list) > 0 :
                    fix_node = torch.tensor(random.choice(tar_list))
                    #fix_node = torch.unique(torch.tensor(sum(random.sample(tar_list, random.choice(range(len(tar_list)))), []))).long()
                    fragnode.append(fix_node)
                    b = torch.zeros_like(fix_node) + i
                    fix_node_batch.append(b)
                    
                    
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            elif sample_num_atoms == 'prior_ref':
                # pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                # ligand_num_atoms = torch.tensor([atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)])
                # ref_ligand_num_atoms = torch.tensor(scatter_sum(torch.ones_like(batch.ligand_element_batch), batch.ligand_element_batch, dim=0).tolist())
                # indicate = ligand_num_atoms<ref_ligand_num_atoms
                # ligand_num_atoms[torch.where(indicate)] = ref_ligand_num_atoms[torch.where(indicate)]
                # ligand_num_atoms = ligand_num_atoms.tolist()
                # batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
                pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = torch.tensor([atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)])
                ref_ligand_num_atoms = torch.tensor(scatter_sum(torch.ones_like(batch.ligand_element_batch), batch.ligand_element_batch, dim=0).tolist())
                comp_ligand_num_atoms = torch.tensor([torch.max(idx).item() if idx.size()[0]>0 else 0 for idx in fragnode])
                indicate = ligand_num_atoms<comp_ligand_num_atoms
                ligand_num_atoms[torch.where(indicate)] = ref_ligand_num_atoms[torch.where(indicate)]
                ligand_num_atoms = ligand_num_atoms + 1
                ligand_num_atoms = ligand_num_atoms.tolist()
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)

            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)
            
            # fragnode = []
            # fix_node_batch = []
            # batch_size = batch.num_graphs
            # for i in range(batch_size):
            #     tar = [inter['Fragment_node'] for inter in batch.IntramolInteraction[i]]
            #     tar_list = []
            #     for item in tar:
            #         if item not in tar_list:
            #             tar_list.append(item)
            #     if len(tar_list) > 0 :
            #         #fix_node = torch.tensor(random.choice(tar_list))
            #         fix_node = torch.unique(torch.tensor(sum(random.sample(tar_list, random.choice(range(len(tar_list)))), []))).long()
            #         fragnode.append(fix_node)
            #         b = torch.zeros_like(fix_node) + i
            #         fix_node_batch.append(b)
                # fix_node = torch.unique(torch.tensor(sum([inter['Fragment_node'] for inter in batch.IntramolInteraction[i] if inter['Fragment_smi'] in list(prep_vocab.keys())], [])))
                # fragnode.append(fix_node)
                # b = torch.zeros_like(fix_node) + i
                # fix_node_batch.append(b)
            
            if frag == 'Frag':        
                fix_node = torch.cat(fragnode, -1).to('cuda')
                fix_node_batch = torch.cat(fix_node_batch, -1).to('cuda')
            elif frag == 'NoFrag':
                fix_node, fix_node_batch = torch.tensor([]).to('cuda'), torch.tensor([]).to('cuda')
            
            batch.fix_node = fix_node
            batch.fix_node_batch = fix_node_batch  
                      
            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                batch = batch,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs')
    parser.add_argument('--frag', type=str, default=False)
    args = parser.parse_args()

    logger = misc.get_logger('sampling')
    os.chdir('./3D-MOL-GENERATION/anonymous')
    # Load config
    config = misc.load_config(args.config)
    #logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    #logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    #logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = PharDiff(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    data = test_set[args.data_id]
    logger.info(f'Inference for \n[{data.protein_filename}][{args.data_id}]')
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms, 
        frag = config.sample.frag
    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list
    }
    logger.info('Sample done!')

    #result_path = args.result_path
    result_path = './' + config.model.checkpoint.split('/')[-3].replace('training', 'outputs')
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))
