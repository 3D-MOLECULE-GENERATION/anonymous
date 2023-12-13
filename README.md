# Pharmacophore-aware pocket based 3D Molecule Generation 
<img src="https://github.com/3D-MOLECULE-GENERATION/anonymous/blob/main/assets/MAIN.png">

## Diffusion process of proposed method
**Note:** For a better understanding of the process, protein pockets have been excluded in the video below.
<table class="center">
<tr>
  <td style="text-align:center;"><b>Forward process</b></td>
  <td style="text-align:center;"><b>Reverse process</b></td>
</tr>
<tr>
  <td><img src="https://github.com/3D-MOLECULE-GENERATION/anonymous/blob/main/assets/sunitinib.gif" width="300"></td>
  <td><img src="https://github.com/3D-MOLECULE-GENERATION/anonymous/blob/main/assets/sunitinib_reverse.gif" width="300"></td>
</tr>
</table>

## Requirements
We include key dependencies below. Our detailed environmental setup is available in [`environment.yml`]
The code has been tested in the following environment:

| Package           | Version   |
|-------------------|-----------|
| Python            | 3.8       |
| PyTorch           | 1.13.1    |
| CUDA              | 11.6      |
| PyTorch Geometric | 2.2.0     |
| RDKit             | 2022.03.2 |

### Install via Conda
```bash
conda create -n PharDiff python=3.8
conda activate PharDiff
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
```
## Data
The data used for training / evaluating the model are organized in the [data](https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link) Google Drive folder.

To train the model from scratch, you need to download the preprocessed lmdb file and split file:
* `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
* `crossdocked_pocket10_pose_split.pt`

To evaluate the model on the test set, you need to download _and_ unzip the `test_set.zip`. It includes the original PDB files that will be used in Vina Docking.

If you want to process the dataset from scratch, you need to download CrossDocked2020 v1.1 from [here](https://bits.csb.pitt.edu/files/crossdock2020/), save it into `data/CrossDocked2020`, and run the scripts in `scripts/data_preparation`:
* [clean_crossdocked.py](scripts/data_preparation/clean_crossdocked.py) will filter the original dataset and keep the ones with RMSD < 1A.
It will generate a `index.pkl` file and create a new directory containing the original filtered data (corresponds to `crossdocked_v1.1_rmsd1.0.tar.gz` in the drive). *You don't need these files if you have downloaded .lmdb file.*
    ```bash
    python scripts/data_preparation/clean_crossdocked.py --source data/CrossDocked2020 --dest data/crossdocked_v1.1_rmsd1.0 --rmsd_thr 1.0
    ```
* [extract_pockets.py](scripts/data_preparation/extract_pockets.py) will clip the original protein file to a 10A region around the binding molecule. E.g.
    ```bash
    python scripts/data_preparation/extract_pockets.py --source data/crossdocked_v1.1_rmsd1.0 --dest data/crossdocked_v1.1_rmsd1.0_pocket10
    ```
* [split_pl_dataset.py](scripts/data_preparation/split_pl_dataset.py) will split the training and test set. We use the same split `split_by_name.pt` as 
[AR](https://arxiv.org/abs/2203.10446) and [Pocket2Mol](https://arxiv.org/abs/2205.07249), which can also be downloaded in the Google Drive - data folder.
    ```bash
    python scripts/data_preparation/split_pl_dataset.py --path data/crossdocked_v1.1_rmsd1.0_pocket10 --dest data/crossdocked_pocket10_pose_split.pt --fixed_split data/split_by_name.pt
    ```

## Training
### Training from scratch
```bash
python scripts/train_diffusion.py configs/training.yml
```
### Trained model checkpoint
comming soon [Google Drive Folder]

## Sampling
### Sampling for pockets in the testset
```bash
python scripts/sample_diffusion.py configs/sampling.yml --data_id {i} # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.
```

### Sampling from pdb file
To sample from a protein pocket (a 10A region around the reference ligand):
```bash
python scripts/sample_for_pocket.py configs/sampling.yml --pdb_path examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb
```

## Evaluation
### Evaluation from sampling results
```bash
python scripts/evaluate_diffusion.py {OUTPUT_DIR} --docking_mode vina_score --protein_root data/test_set
```
The docking mode can be chosen from {qvina, vina_score, vina_dock, none}

Note: It will take some time to prepare pqdqt and pqr files when you run the evaluation code with vina_score/vina_dock docking mode for the first time.


## Result
The main results for the proposed model are presented in the table below. For a more comprehensive overview of the results obtained with our model, please refer to the manuscript.

### Property of Generated Molecule
| Model      | VinaScore  | VinaMin   | VinaDock   | HighAiffinity | 
|------------|------------|-----------|------------|---------------|
| LiGAN      |    -  |    -   | -6.33  |  0.21  | 
| GraphBP    |    -  |    -   | -4.80  |  0.14  | 
| AR         | -5.75 | -6.18 | -6.75  |  0.20  |
| Pocket2Mol | -5.15 | -6.42 | -7.15  |  0.14  | 
| TargetDiff | -5.47 | -6.64 | -7.80  |  0.31  | 
| Ours       | -6.64 | -7.20 | -8.16  |  0.47  | 
| *Reference*  | -6.36 | -6.71 | -7.45  |  -  | 

### Distribution of RMSD before and after Docking
<img src="https://github.com/3D-MOLECULE-GENERATION/anonymous/blob/main/assets/DifferenceRMAD.png" width="400">

### Demo video of Molecular Dynamics about Generated Molecule
<img src="https://github.com/3D-MOLECULE-GENERATION/anonymous/blob/main/assets/MD_result.gif" width="400">
