import os
import re
import shutil
import argparse
import traceback

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from models.search import beam_search
from utils.transforms import *
from utils.train import *
from utils.misc import get_new_log_dir

from tqdm import tqdm
from operator import itemgetter

os.environ["TOKENIZERS_PARALLELISM"] = "False"
from models.TranMResPro import MolT5EmbeddingModule, TranMResProModel
from utils.early_stop import *

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def load_checkpoint(ckpt_path, model, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    return model, start_epoch, best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_res.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--k', type=int, help='number of retrival molecules about per input molecule', default=10)
    parser.add_argument('--batch_size', type=int, help='number of batch size', default=1)
    # parser.add_argument('--ckpt_path', type=str,
    #                     default='./logs/train_res_2025_03_10__22_26_51/checkpoints/model_epoch19.pt')
    parser.add_argument('--ckpt_path', type=str,
                        default='./logs/train_res_2024_09_13__10_13_57/checkpoints/model_epoch50.pt')

    args = parser.parse_args()
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')

    args.out = '.'.join(args.ckpt_path.split('.')[:-1]) + '_' + 'generate_smiles_top10_valid.csv'
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

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

    # Datasets and loaders
    logger.info('Loading dataset...')
    subsets = get_dataset(
        config=config,
        transform=transform,
    )

    collate_exclude_keys = ['ligand_nbh_list']
    train_set, val_set, test_set = subsets['train'], subsets['valid'], subsets['test']

    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, exclude_keys=collate_exclude_keys)
    print('train_loader process done!')

    test_interator = inf_iterator(test_loader)

    logger.info('Building model...')
    model = TranMResProModel(
        config.model,
        protein_res_feature_dim=(27, 3),
        ligand_atom_feature_dim=(13, 1),
        num_props=config.train.num_props,
    ).to(args.device)

    # load checkpoint
    model, start_epoch, best_loss = load_checkpoint(args.ckpt_path, model, args.device)

    # retrival
    with open(r'data/retrival_database/retrieval_topsim100_chembl.pkl', 'rb') as f:
        train_retrieval = pickle.load(f)

    # Precompute and cache MolT5 embeddings
    molT5_embedding_module = MolT5EmbeddingModule(model_path="models/molT5").to(args.device)
    molT5_embedding_cache = {}
    with torch.no_grad():
        for smile, retrieval_list in tqdm(train_retrieval.items()):
            molT5_embedding_cache[smile] = retrieval_list[:args.k]


    def gen():
        model.eval()
        df3 = pd.DataFrame()
        batch_protein_name = []
        for batch in tqdm(test_loader):
            batch = batch.to(args.device)
            batch_size = len(batch)

            smile_list = batch['ligand_smile']

            # retrival data
            batch_retricval_list = itemgetter(*smile_list)(molT5_embedding_cache)
            with torch.no_grad():
                ret_embeddings = molT5_embedding_module(batch_retricval_list, args.device)
            batch_ret_embeddings = ret_embeddings.reshape(len(smile_list), args.k, -1)

            # generate molecules
            num_beams = 200
            topk = 200
            gen_num = 10
            if config.train.num_props:
                prop = torch.tensor([config.generate.prop for i in range(batch_size * num_beams)],
                                    dtype=torch.float).to(args.device)
                assert prop.shape[-1] == config.train.num_props
                num = int(bool(config.train.num_props))
            else:
                num = 0
                prop = None
            generate = []

            beam_output = beam_search(model, config.model.decoder.smiVoc, num_beams,
                                      batch_size, config.model.decoder.tgt_len + num, topk, batch,
                                      batch_ret_embeddings,
                                      prop, args.device)
            beam_output = beam_output.view(batch_size, topk, -1)
            for i, item in enumerate(beam_output):
                for j in item:
                    if len(generate) == gen_num:
                        break
                    smile = [config.model.decoder.smiVoc[n.item()] for n in j.squeeze()]
                    smile = re.sub('[&$^]', '', ''.join(smile))
                    if Chem.MolFromSmiles(smile) is None:
                        continue
                    else:
                        generate.append(smile)

                df1 = pd.DataFrame([batch.protein_filename] * len(generate), columns=['PROTEINS'])
                df2 = pd.DataFrame(generate, columns=['SMILES'])
                df3 = df3._append(pd.concat([df1, df2], join='outer', axis=1))
            print(f'protein {batch.protein_filename} has {len(generate)} molecules')
            df3.to_csv(args.out, index=False)

            # batch_protein_name.extend(batch.protein_filename)
            # batch_beam_output.append(beam_output)

        # batch_beam_output = torch.cat(batch_beam_output).cpu().numpy()

        # batch_smi = []
        # for i, item in enumerate(batch_beam_output):
        #     generate = []
        #     for j in item:
        #         smile = [config.model.decoder.smiVoc[n.item()] for n in j.squeeze()]
        #         smile = re.sub('[&$^]'
        #                        , '', ''.join(smile))
        #         generate.append(smile)
        #     batch_smi.append(generate)
        #
        #     df1 = pd.DataFrame([batch_protein_name[i]] * topk, columns=['PROTEINS'])
        #     df2 = pd.DataFrame(generate, columns=['SMILES'])
        #     df3 = pd.concat([df1, df2], join='outer', axis=1)
        # df3.to_csv(args.out, index=False)


    try:
        gen()
    except Exception as e:
        traceback.print_exc()
