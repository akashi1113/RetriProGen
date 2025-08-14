import json
import os.path

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from test_mol.metrics import Estimate_QED, Estimate_SA
from utils.docking import QVinaDockingTask
from easydict import EasyDict as edic
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def update_dict(data, key, value):
    if key in data:  # 判断key是否已存在dic中，存在则向键中添加值，不存在则直接添加键值对
        if isinstance(data[key], list):  # 判断该键对应的值是否多个
            thelist = data[key]
            thelist.append(value)
            data[key] = thelist
        else:
            thelist = data[key]
            data[key] = list()
            data[key].append(thelist)
            data[key].append(value)
    else:
        data[key] = [value]
    return data

# input
epoch = 50
# path = f'./logs/train_res_2025_03_10__22_26_51/checkpoints/model_epoch{epoch}_generate_smiles_top10_valid.csv'
path = f'./logs/train_res_2024_09_13__10_13_57/checkpoints/model_epoch{epoch}_generate_smiles_top10_valid.csv'
df = pd.read_csv(path)
# df.dropna(axis=0, how='any', inplace=True)
protein_path = '/workspace/kangchenglong/drug_design_protein/data/crossdocked_pocket10/'
#protein_path = '/home/aichengwei/code/own_code/drug_design_protein/data/crossdocked_pocket10/'


# with open(f'./logs/train_res_2024_06_11__15_51_14/checkpoints/model_epoch{epoch}_generate_smiles_top10_valid.json',
#           'r') as f:
#     info_data = json.load(f)
# 
# qed = []
# sa = []
# affi = []
# for key, value in info_data.items():
#     protein_name = key
#     gen_prop = value
#     for gen_index in gen_prop:
#         if gen_index[2] != 0:
#             qed.append(gen_index[0])
#             sa.append(gen_index[1])
#             affi.append(gen_index[2])
            

# from scrach
vina_dict = {}
qed_list, sa_list = [], []
aff_list = []
for index, row in df.iterrows():
    protein_name = row['PROTEINS'].split('.')[0]
    ligand_name = protein_name.split('_pocket10')[0] + '.sdf'
    smiles = row['SMILES']
    try:
        mol = Chem.MolFromSmiles(smiles)
        pdb_block = os.path.join(protein_path, protein_name)
        lgaind_pdb_block = os.path.join(protein_path, ligand_name)
        vina_task = QVinaDockingTask(pdb_block, mol, lgaind_pdb_block, use_uff=True)

        vina_score = vina_task.run_sync()
        affi = vina_score
        qed = Estimate_QED(smiles)
        sa = Estimate_SA(smiles)
    except:
        mol = None
        affi = 0
        qed = 0
        sa = 0
    # cal QED

    qed_list.append(qed)
    sa_list.append(sa)
    aff_list.append(affi)

    update_dict(vina_dict, protein_name, [qed, sa, affi])

    print(f'processing {index + 1} / {len(df)} data')

info_json = json.dumps(vina_dict, sort_keys=False, indent=4, separators=(',', ': '))

with open('.'.join(path.split('.')[:-1]) + '.json', 'w') as f:
    f.write(info_json)

print('model at epoch {} results: affi_aver: {}, qed aver: {}, sa aver: {}'.format(epoch, np.mean(aff_list),
                                                                                   np.mean(qed_list), np.mean(sa_list)))
print('...')
# prepare p
