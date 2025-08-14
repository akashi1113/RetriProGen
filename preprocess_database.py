from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm

from test_mol.metrics import calc_logp, calc_qed, calc_sas, Estimate_logP, Estimate_QED, Estimate_SA

def preprocess_smiles(chembl_smiles):
    chembl_data = pd.read_csv(f'data/chembl34/chembl_34_chemreps.txt', sep='\t')
    chembl_smiles = chembl_data['canonical_smiles'].tolist()

    print('process logp')
    with Pool(processes=32) as pool:
        log_p = list(tqdm(pool.imap(Estimate_logP, chembl_smiles), total=len(chembl_smiles)))

    print('process qed')
    with Pool(processes=32) as pool:
        qed = list(tqdm(pool.imap(Estimate_QED, chembl_smiles), total=len(chembl_smiles)))

    print('process sa')
    with Pool(processes=32) as pool:
        sa = list(tqdm(pool.imap(Estimate_SA, chembl_smiles), total=len(chembl_smiles)))

    chembl_data['log_p'] = log_p
    chembl_data['qed'] = qed
    chembl_data['sas'] = sa

    chembl_data.to_csv('data/chembl34/chembl_smiles_prop.csv', index=False)

chembl_data = pd.read_csv(f'data/chembl34/chembl_smiles_prop.csv')


print('process filter qed')
attr = 'qed'
chembl_data_f = chembl_data[(chembl_data['qed'] >= 0.9]

print('process filter sa')
attr = 'sas'
chembl_data_f = chembl_data_f.sort_values(by=attr, ascending=True)[:5000]
# aver_sa = np.mean(chembl_data[attr])
# res = chembl_data.sort_values(by=attr, ascending=True)
# chembl_data_f = chembl_data_f[chembl_data_f[attr] <= aver_sa]

chembl_data_f.to_csv('data/chembl34/chembl_smiles_prop_qed09_saLow5000.csv', index=False)


