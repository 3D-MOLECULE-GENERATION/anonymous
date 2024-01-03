import sys
sys.path.append("./3D-MOL-GENERATION/anonymous")
sys.path.append("./3D-MOL-GENERATION/anonymous/datasets")
from mol_tree import *
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import glob
from utils.data import PDBProtein, parse_sdf_file
from pl_data import ProteinLigandData, torchify_dict
from plip.structure.preparation import PDBComplex
import traceback

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        no_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    interactions = self.get_interaction(data_prefix, ligand_fn)
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    
                    residue_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_residue()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    interactions_dict = {"IntramolInteraction":interactions}
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                        residue_dict=torchify_dict(residue_dict),
                        inter_dict=torchify_dict(interactions_dict),
                    )
                    #data['intramolInteraction'] = interactions
                    # 
                    #data.intramolInteraction = interactions

                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                    no_skipped += 1
                except Exception as error:
                    num_skipped += 1
                    print('Skipping (%d/%d) %s' % (num_skipped,num_skipped + no_skipped, ligand_fn, ))
                    print(error, '\n')
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data
        
    def extract_name(self, classname):
        return classname.split('.')[-1].split("'")[0]    
    
    def get_interaction(self, data_prefix, ligand_fn):
        # complex_1m4n_A_rec_1m7y_ppg_lig_tt_min_0.pdb
        complex_path = data_prefix.replace("crossdocked_v1.1_rmsd1.0_pocket10", "complex")
        complex_fn = ligand_fn.replace(".sdf", "_pocket10_complex")
        c_p = os.path.join(complex_path, complex_fn)
        pdb = glob.glob(c_p + "/*pdb")
       
        mol = PDBComplex()
        mol.load_pdb(pdb[0])
        mol.analyze() 

        longnames = [x.longname for x in mol.ligands]
        bsids = [":".join([x.hetid, x.chain, str(x.position)]) for x in mol.ligands]
        indices = [j for j,x in enumerate(longnames) if x == 'ISK']        
        
        inter_list = [] 
        for bs in bsids:
            interactions = mol.interaction_sets[bs]
            for inter in interactions.all_itypes:
                inter_list = self.parse_interaction(inter, inter_list, mol)
                
        frag_vocab_dir = './3D-MOL-GENERATION/anonymous/datasets/frag_vocab.pickle' 
        with open(frag_vocab_dir, 'rb') as fr:
            Frag_vocab = pickle.load(fr)  
        vocab_idx_dic = dict(zip(list(Frag_vocab.keys()), np.arange(len(list(Frag_vocab.keys())))))         
        vocab_dir = './3D-MOL-GENERATION/anonymous/datasets/vocab.txt'
        with open(vocab_dir, 'r') as f:
            vocab = [x.strip() for x in f.readlines()]    
        reference_vocab = np.load('./3D-MOL-GENERATION/anonymous/utils/reference.npy', allow_pickle=True).item()  
        vocab = Vocab(vocab)           
        temp = pdb[0].split('/')
        temp_dir = os.path.join('/'.join(temp[:-2]), temp[-1].replace("complex_", "").replace(".pdb", ".sdf"))
        crossdock_dir_sdf = temp_dir.replace('complex', 'crossdocked_v1.1_rmsd1.0_pocket10')
        suppl = Chem.SDMolSupplier(crossdock_dir_sdf)
        mol_list = [x for x in suppl if x is not None]  
        for i, mol in enumerate(mol_list):
            tree_list = []
            jt = MolTree(mol, reference_vocab)
            tree_list.append(jt)  
            pairbynode_smi = []
            
            
            preprocessed_inter_list = []
            for inter in inter_list:
                interbynode=[]
                for node in jt.nodes:                
                    if (inter['interaction'] == 'hydroph_interaction' or 'halogenbond' or 'hbond'):
                        if (inter["sdf_idx"] in node.clique) == True:
                            interbynode.append(node.clique)                           
                    if inter['interaction'] == 'pistack':
                        if (sorted(inter["sdf_idx"]) == sorted(node.clique)) == True:
                            interbynode.append(node.clique)                            
                    if inter['interaction'] == 'pication':
                        if len(inter["sdf_idx"]) > 1:
                            if (sorted(inter["sdf_idx"]) == sorted(node.clique)) == True:
                                interbynode.append(node.clique)                 
                        else:
                            if (inter["sdf_idx"][0] in node.clique) == True:
                                interbynode.append(node.clique)  
                
                try:
  
                    if not len(interbynode) == 0: 
                        a = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(jt.mol, sum(interbynode, []), kekuleSmiles=True))
                        smi = Chem.MolFragmentToSmiles(jt.mol, sum(interbynode, []), kekuleSmiles=True)

                        inter['Fragment_smi'] = smi
                        inter['Fragment_idx'] = vocab_idx_dic[smi]
                        inter['Fragment_node'] = list(set(sorted(sum(interbynode, [])))) 
                        if (inter['interaction'] == 'hydroph_interaction'):
                            inter['interaction_onehot'] = [1,0,0,0,0]
                            extract_inter = {key: value for key, value in inter.items() if (key == 'restype') 
                                                                                                or (key == 'resnr')
                                                                                                or (key == 'interaction')
                                                                                                or (key == 'sdf_idx')
                                                                                                or (key == 'Fragment_smi')
                                                                                                or (key == 'Fragment_idx')
                                                                                                or (key == 'Fragment_node')
                                                                                                or (key == 'interaction_onehot')}                            
                        if (inter['interaction'] == 'halogenbond'):
                            inter['interaction_onehot'] = [0,1,0,0,0]
                            extract_inter = {key: value for key, value in inter.items() if (key == 'restype') 
                                                                                                or (key == 'resnr')
                                                                                                or (key == 'interaction')
                                                                                                or (key == 'sdf_idx')
                                                                                                or (key == 'Fragment_smi')
                                                                                                or (key == 'Fragment_idx')
                                                                                                or (key == 'Fragment_node')
                                                                                                or (key == 'interaction_onehot')}   
                        if (inter['interaction'] == 'hbond'):
                            inter['interaction_onehot'] = [0,0,1,0,0]
                            extract_inter = {key: value for key, value in inter.items() if (key == 'restype') 
                                                                                                or (key == 'resnr')
                                                                                                or (key == 'interaction')
                                                                                                or (key == 'sdf_idx')
                                                                                                or (key == 'Fragment_smi')
                                                                                                or (key == 'Fragment_idx')
                                                                                                or (key == 'Fragment_node')
                                                                                                or (key == 'interaction_onehot')} 
                            
                        if (inter['interaction'] == 'pistack'):
                            inter['interaction_onehot'] = [0,0,0,1,0]
                            extract_inter = {key: value for key, value in inter.items() if (key == 'restype') 
                                                                                                or (key == 'resnr')
                                                                                                or (key == 'interaction')
                                                                                                or (key == 'sdf_idx')
                                                                                                or (key == 'Fragment_smi')
                                                                                                or (key == 'Fragment_idx')
                                                                                                or (key == 'Fragment_node')
                                                                                                or (key == 'interaction_onehot')} 
                                                        
                        if (inter['interaction'] == 'pication'):
                            inter['interaction_onehot'] = [1,0,0,0,1]  
                            extract_inter = {key: value for key, value in inter.items() if (key == 'restype') 
                                                                                                or (key == 'resnr')
                                                                                                or (key == 'interaction')
                                                                                                or (key == 'sdf_idx')
                                                                                                or (key == 'Fragment_smi')
                                                                                                or (key == 'Fragment_idx')
                                                                                                or (key == 'Fragment_node')
                                                                                                or (key == 'interaction_onehot')} 
                                                
                        preprocessed_inter_list.append(extract_inter)    
                except Exception as e:
                    print(e)
                    print("ERROR")
                    err_msg = traceback.format_exc()
                    print(err_msg)
            if len(preprocessed_inter_list) == 0:
                raise Exception("No interaction")

        return preprocessed_inter_list

            
    def parse_interaction(self, inter, inter_list, mol):
        
        if (self.extract_name(str(inter.__class__)) == 'hydroph_interaction') == True:
            inter_dict = inter._asdict()
            inter_dict["interaction"] = self.extract_name(str(inter.__class__))
            inter_dict["sdf_idx"] = inter.ligatom_orig_idx - 1 - min(mol.ligands[0].can_to_pdb.values())
            inter_list.append(inter_dict)  
             
        if self.extract_name(str(inter.__class__)) == 'hbond':
            inter_dict = inter._asdict()
            inter_dict["interaction"] = self.extract_name(str(inter.__class__))
            if inter.d_orig_idx > min(mol.ligands[0].can_to_pdb.values()):
                inter_dict["sdf_idx"] = inter.d_orig_idx -1 - min(mol.ligands[0].can_to_pdb.values())
            else:
                inter_dict["sdf_idx"] = inter.a_orig_idx -1 - min(mol.ligands[0].can_to_pdb.values())    
            inter_list.append(inter_dict)   
            
        if self.extract_name(str(inter.__class__)) == 'pistack':
            inter_dict = inter._asdict()
            inter_dict["interaction"] = self.extract_name(str(inter.__class__))
            inter_dict["sdf_idx"] = [idx - 1 - min(mol.ligands[0].can_to_pdb.values()) for idx in inter_dict['ligandring'].atoms_orig_idx]
            inter_list.append(inter_dict)  
            
        if self.extract_name(str(inter.__class__)) == 'halogenbond':
            inter_dict = inter._asdict()
            inter_dict["interaction"] = self.extract_name(str(inter.__class__))
            inter_dict["sdf_idx"] = inter.don.x_orig_idx -1 - min(mol.ligands[0].can_to_pdb.values())
            inter_list.append(inter_dict)       
            
        if self.extract_name(str(inter.__class__)) == 'pication':
            inter_dict = inter._asdict()
            inter_dict["interaction"] = self.extract_name(str(inter.__class__))        
        
            # charge쪽이 ligand
            if min(inter.charge.atoms_orig_idx) > min(mol.ligands[0].can_to_pdb.values()):
                inter_dict["sdf_idx"] = [idx -1 - min(mol.ligands[0].can_to_pdb.values()) for idx in inter_dict['charge'].atoms_orig_idx]
            
            # ring쪽이 ligand 
            else:
                inter_dict["sdf_idx"] = [idx -1 - min(mol.ligands[0].can_to_pdb.values()) for idx in inter_dict['ring'].atoms_orig_idx]
            inter_list.append(inter_dict)  
        
        return inter_list              
                 
             
        
if __name__ == '__main__':
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str)
    # args = parser.parse_args()
    path = './3D-MOL-GENERATION/anonymous/data/crossdocked_v1.1_rmsd1.0_pocket10'
    dataset = PocketLigandPairDataset(path)
    print(len(dataset), dataset[0])
    for idx in range(len(dataset)):
        print(dataset[idx].protein_filename)
        print(dataset[idx].ligand_filename)
