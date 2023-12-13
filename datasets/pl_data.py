import torch
import torch_scatter
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type', 'amino_acid', 'protein_atom2residue')


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, residue_dict=None, inter_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item
                
        if residue_dict is not None:
            for key, item in residue_dict.items():
                instance[key] = item
                
        if inter_dict is not None:
            for key, item in inter_dict.items():
                instance[key] = item
        
        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                                                  if instance.ligand_bond_index[0, k].item() == i]
                                       for i in instance.ligand_bond_index[0]}
        
        """
        Here it for "residue_pos"
        """
        instance['residue_pos'] = torch.stack([instance['pos_CA'], 
                                               instance['pos_C'], 
                                               instance['pos_N'], 
                                               instance['pos_O']], dim=1)
        
        
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        else:
            return super().__inc__(key, value)


class ProteinLigandDataLoader(DataLoader):

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def get_batch_connectivity_matrix(ligand_batch, ligand_bond_index, ligand_bond_type, ligand_bond_batch):
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []
    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]
        bond_type = ligand_bond_type[ligand_bond_batch == batch_index]
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.
        connectivity_matrix = torch.zeros(batch_ligand_size[batch_index], batch_ligand_size[batch_index],
                                          dtype=torch.int)
        for s, e, t in zip(start_index, end_index, bond_type):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
        batch_connectivity_matrix.append(connectivity_matrix)
    return batch_connectivity_matrix


def collate_mols(mol_dicts):
    data_batch = {}
    batch_size = len(mol_dicts)
    for key in ['protein_pos', 'protein_atom_feature','ligand_pos','res_idx','amino_acid','IntramolInteraction']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
        
    data_batch['residue_pos'] = \
        torch.cat([torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).unsqueeze(0) for key in ['pos_N', 'pos_CA', 'pos_C', 'pos_O']], dim=0).permute(1,0,2)
        
    repeats = torch.cat([mol_dict['protein_atom2residue'].bincount() for mol_dict in mol_dicts])
    data_batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)  
    
    for key in ['protein_element', 'amino_acid'] :
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)       
    print("++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++")
    
    return data_batch    
      