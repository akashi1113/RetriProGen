import argparse

import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from multiprocessing import Pool
# from DeepPurpose import utils, dataset
# from DeepPurpose import DTI as models
from functools import partial
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import pickle
from Bio import PDB
import os

from rdkit.Contrib.SA_Score import sascorer
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from utils.datasets import get_dataset
from utils.misc import get_new_log_dir, get_logger, load_config
from utils.transforms import FeaturizeLigandAtom, get_mask, Res2AtomComposer, EdgeSample, ContrastiveSample, RefineData, \
    LigandCountNeighbors, FocalBuilder


def tanimoto_dist_func(lead_fp, ret):
    return DataStructs.TanimotoSimilarity(
        lead_fp,
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ret), 2, nBits=2048))


parser = PDB.PDBParser()
cout_e1 = 0
cout_e2 = 0
cout_e3 = 0
cout_e4 = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_res.yml')
    parser.add_argument('--logdir', type=str, default='./logs_preprocess_data/')
    parser.add_argument('--data_type', type=str, default='train')
    args = parser.parse_args()

    config = load_config(args.config)

    # initialize logger
    log_dir = get_new_log_dir(args.logdir)
    logger = get_logger('train', log_dir)

    # load training data
    logger.info('Loading crossdock dataset...')

    ligand_featurizer = FeaturizeLigandAtom()
    masking = get_mask(config.train.transform.mask)
    composer = Res2AtomComposer(27, ligand_featurizer.feature_dim, config.model.encoder.knn)
    edge_sampler = EdgeSample(config.train.transform.edgesampler)
    cfg_ctr = config.train.transform.contrastive
    contrastive_sampler = ContrastiveSample(cfg_ctr.num_real, cfg_ctr.num_fake, cfg_ctr.pos_real_std,
                                            cfg_ctr.pos_fake_std, config.model.field.knn)

    transform = Compose([
        RefineData(),
        LigandCountNeighbors(),
        ligand_featurizer,
        masking,
        composer,
        FocalBuilder(),
        edge_sampler,
        contrastive_sampler,
    ])

    subsets = get_dataset(
        config=config,
        transform=transform,
    )
    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(subsets[args.data_type], batch_size=1, exclude_keys=collate_exclude_keys, )

    # load Chembl data
    logger.info('Loading chembl dataset...')
    data = pd.read_csv(r'chembl_34_chemreps.txt', sep='\t')
    chembl_smiles = data['canonical_smiles']

    # 为训练集构建一个一对多映射的检索字典，根据训练集的ligand_smile去匹配data_attr中对应的10个smiles
    train_retrieval = {}
    base_path = r'crossdocked_pocket10'
    drug_encoding, target_encoding = 'MPNN', 'CNN'

    for lead in train_set[10]:
        sims = list(map(
            partial(
                tanimoto_dist_func,
                AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(lead.ligand_smile), 2, nBits=2048)),
            data.smiles.tolist()))
        smile = lead.ligand_smile
        mol = Chem.MolFromSmiles(smile)
        plogp = Descriptors.MolLogP(mol) - sascorer.calculateScore(mol)
        data['similarity'] = sims
        data_for_this_lead = data.sort_values(by='similarity', ascending=False)  # 按相似度降序排序
        pdb_file_name = lead.protein_filename
        pdb_path = os.path.join(base_path, pdb_file_name)
        structure = parser.get_structure('2FH7', pdb_path)
        ppb = PDB.PPBuilder()
        # 初始化一个空字符串来保存蛋白质序列
        protein_sequence = ""
        # 遍历模型中的所有链
        for model in structure:
            for chain in model:
                # 使用PPBuilder构建多肽
                for pp in ppb.build_peptides(chain):
                    # 获取多肽的序列并添加到蛋白质序列字符串中
                    protein_sequence += pp.get_sequence()
        X_target = [protein_sequence] * 1000

        y = [10] * 1000
        X_pred = utils.data_process(X_drug, X_target, y,
                                    drug_encoding, target_encoding,
                                    split_method='no_split')
        y_pred = net.predict(X_pred)
        data_for_this_lead['predict_score'] = y_pred
        y_1 = [10]
        lead.ligand_smile = str(lead.ligand_smile)
        protein_sequence = str(protein_sequence)
        X_drug_1 = [lead.ligand_smile]
        X_target_1 = [protein_sequence]
        X_pred_1 = utils.data_process(X_drug_1, X_target_1, y_1,
                                      drug_encoding, target_encoding,
                                      split_method='no_split')
        y_pred_1 = net.predict(X_pred_1)
        print(y_pred_1)

    # C[C@]12CCc3c(ccc4cc(O)ccc34)[C@@H]1CCC2=O 这是输入分子
    # CC[C@@H](N)c1ccc(Cl)c(Oc2ccccc2)c1F,CSc1cccc(C2=C3C=CC(F)=CC3=C(CC(=O)O)=C2C)c1,CSc1cccc(C2=C3C=CC(F)=CC3=C2C(=O)N=C2)cc1,CSc1cccc(-c2nn(C)c(F)ccc2-c2cccnc2)c1,Cc1noc(C)c1-c1ccc2c(c1)[C@H](c1ccccc1)NC2=O,Oc1cc(-c2ccc(F)cc2)cc2cccnc12,CSc1cccc(-c2nn(C)c(F)cc2-c2ccccc2)c1,CSc1cccc(-c2cccc3ccc(O)cc3o2)c1,CSc1cccc(-c2nc(C)n[nH]2)cc1-c1ccc(F)cc1,CSc1cccc(-c2nc(C)nc(-c3cccs3)n2)c1
    # 10_3dzh_A_rec_3u4i_cvr_lig_tt_docked_0_pocket10.pdbqt 这是蛋白口袋

    #     count = 0
    #     #集合里可能配体分子的smiles是相同的所以得区分开来
    #     if lead.ligand_smile in train_retrieval:
    #         length=len(train_retrieval[lead.ligand_smile])
    #         count=length
    #     for index, element in data_for_this_lead.iterrows():
    #         if count >= 10:
    #             print("saved")
    #             break  # 如果已经保存了10个分子，则退出循环
    #         if(element['qed']>lead.ligand_qed and element['logp-sa']>plogp and y_pred_1[0]<=element['predict_score']):
    #             if lead.ligand_smile not in train_retrieval:
    #                 train_retrieval.setdefault(lead.ligand_smile, []).append(element['smiles'])
    #                 count += 1
    #             elif element['smiles'] not in train_retrieval[lead.ligand_smile]:
    #                 train_retrieval[lead.ligand_smile].append(element['smiles'])
    #                 count += 1
    #             if count == 10:
    #                 cout_e1+=1
    #                 print("saved")
    #                 break
    #
    #     if count < 10:
    #         for index, element in data_for_this_lead.iterrows():
    #             if count >=10:
    #                 print("saved")
    #                 break  # 如果已经保存了10个分子，则退出循环
    #             if (element['qed'] > lead.ligand_qed and y_pred_1[0]<= element['predict_score']):
    #                 if lead.ligand_smile not in train_retrieval:
    #                     train_retrieval.setdefault(lead.ligand_smile, []).append(element['smiles'])
    #                     count += 1
    #                 elif element['smiles'] not in train_retrieval[lead.ligand_smile]:
    #                     train_retrieval[lead.ligand_smile].append(element['smiles'])
    #                     count += 1
    #                 if count == 10:
    #                     cout_e2 += 1
    #                     print("saved")
    #                     break
    #     if count < 10:
    #         for index, element in data_for_this_lead.iterrows():
    #             if count >=10:
    #                 print("saved")
    #                 break  # 如果已经保存了10个分子，则退出循环
    #             if (element['qed'] > lead.ligand_qed):
    #                 if lead.ligand_smile not in train_retrieval:
    #                     train_retrieval.setdefault(lead.ligand_smile, []).append(element['smiles'])
    #                     count += 1
    #                 elif element['smiles'] not in train_retrieval[lead.ligand_smile]:
    #                     train_retrieval[lead.ligand_smile].append(element['smiles'])
    #                     count += 1
    #                 if count == 10:
    #                     cout_e3 += 1
    #                     print("saved")
    #                     break
    #     if count < 10:
    #         for index, element in data_for_this_lead.iterrows():
    #             if count >=10:
    #                 print("saved")
    #                 break  # 如果已经保存了10个分子，则退出循环
    #             if lead.ligand_smile not in train_retrieval:
    #                 train_retrieval.setdefault(lead.ligand_smile, []).append(element['smiles'])
    #                 count += 1
    #             elif element['smiles'] not in train_retrieval[lead.ligand_smile]:
    #                 train_retrieval[lead.ligand_smile].append(element['smiles'])
    #                 count += 1
    #             if count>=10:
    #                 cout_e4+=1
    #                 print("saved")
    #                 break  # 如果已经保存了10个分子，则退出循环
    #
    # #保存每个数据retrieval的结果
    # print("cout_e1:",cout_e1)
    # print("cout_e2:",cout_e2)
    # print("cout_e3:",cout_e3)
    # print("cout_e4:",cout_e4)
    # with open('retrieval_allfe_alldata_predictscore.pkl', 'wb') as f:
    #     pickle.dump(train_retrieval, f)


if __name__ == '__main__':
    main()
