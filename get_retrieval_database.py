import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from multiprocessing import Pool
from functools import partial
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import pickle
from Bio import PDB
import os

from rdkit.Contrib.SA_Score import sascorer
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
from pandarallel import pandarallel
from utils.datasets import get_dataset
from utils.misc import get_new_log_dir, get_logger, load_config
from utils.transforms import FeaturizeLigandAtom, get_mask, Res2AtomComposer, EdgeSample, ContrastiveSample, RefineData, \
    LigandCountNeighbors, FocalBuilder
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
def tanimoto_dist_func(items):
    (lead_smile, chembl_smile) = items
    lead_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(lead_smile), 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(
        lead_fp,
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(chembl_smile), 2, nBits=2048))


def tanimoto_dist_fun_mulcpu(args):
    return tanimoto_dist_func(*args)


parser = PDB.PDBParser()
cout_e1 = 0
cout_e2 = 0
cout_e3 = 0
cout_e4 = 0

from pandarallel import pandarallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_res.yml')
    parser.add_argument('--logdir', type=str, default='./logs_preprocess_data/')
    parser.add_argument('--data_type', type=str, default='train')

    parser.add_argument('--retrival_num', type=int, default=100)
    parser.add_argument('--num_cpu', type=int, default=32)

    args = parser.parse_args()

    config = load_config(args.config)

    # initialize logger
    log_dir = get_new_log_dir(args.logdir)
    logger = get_logger('train', log_dir)

    args.num_cpu = os.cpu_count() - 24
    logger.info(f'num_cpu is {args.num_cpu}...')

    # load training data
    logger.info('Loading crossdock dataset...')
    with open('data/crossdocked_pocket10/index.pkl', 'rb') as f:
        index = pickle.load(f)
    smiles = []
    for i, (pocket_fn, ligand_fn, _, rmsd_str) in tqdm(enumerate(tqdm(index))):
        try:
            mol = Chem.MolFromMolFile(os.path.join('data/crossdocked_pocket10/', ligand_fn))
            smiles.append(Chem.MolToSmiles(mol))
        except Exception as e:
            continue
    unique_smiles = list(set(smiles))

    # load Chembl data
    logger.info('Loading chembl dataset...')
    chembl_data = pd.read_csv(f'data/chembl34/chembl_smiles_prop_qed09_saLow5000.csv')
    chembl_smiles = chembl_data['canonical_smiles']

    # pre-processing data
    logger.info('calculating sim score...')
    input = list(itertools.product(unique_smiles, chembl_smiles.tolist()))
    #input = pd.DataFrame(input, columns=['smile'])
    input = pd.DataFrame(input, columns=['lead_smile', 'chembl_smile'])

    # score= input['smile'].parallel_apply(
    #      lambda x: tanimoto_dist_func(x))
    # input['score']=score
    # score = input.parallel_apply(lambda x: tanimoto_dist_func((x['lead_smile'], x['chembl_smile'])), axis=1)
    # input['score'] = score

    # with Pool(processes=args.num_cpu) as pool:
    #     result_sim = list(tqdm(pool.imap(tanimoto_dist_func, input), total=len(input)))
    with Pool(processes=args.num_cpu) as pool:
        result_sim = list(tqdm(pool.imap(tanimoto_dist_func, input.itertuples(index=False, name=None)), total=len(input)))

    # propress data
    train_retrieval = {}
    for index in range(len(unique_smiles)):
        smile = unique_smiles[index]
        sims = result_sim[index * len(chembl_smiles):(index + 1) * len(chembl_smiles)]

        # sort by sim
        sort_idx = np.argsort(-np.array(sims))
        # sort_sim = [sims[i] for i in sort_idx]

        # retrival smiles
        retrival_smiles = chembl_smiles[sort_idx][:args.retrival_num]

        train_retrieval[smile] = retrival_smiles.tolist()

    logger.info('saving retrieval data...')

    if not os.path.exists(f'data/retrival_database'):
        os.makedirs(f'data/retrival_database')

    with open('data/retrival_database/retrieval_topsim100_chembl.pkl', 'wb') as f:
        pickle.dump(train_retrieval, f)


if __name__ == '__main__':
    main()
