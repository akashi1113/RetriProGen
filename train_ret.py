import os
import shutil
import argparse
import traceback
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from utils.transforms import *
from utils.train import *
from utils.misc import get_new_log_dir

from tqdm import tqdm
from operator import itemgetter

os.environ["TOKENIZERS_PARALLELISM"] = "False"
from models.TranMResPro import MolT5EmbeddingModule, TranMResProModel
from utils.early_stop import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def load_checkpoint(ckpt_path, model, optimizer, scheduler, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    return model, optimizer, scheduler, start_epoch, best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_res.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--k', type=int, help='number of retrival molecules about per input molecule', default=10)
    parser.add_argument('--batch_size', type=int, help='number of batch size', default=32)

    args = parser.parse_args()
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
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

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, exclude_keys=collate_exclude_keys,
                              drop_last=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, exclude_keys=collate_exclude_keys, drop_last=True)
    print('train_loader process done!')

    train_interator = inf_iterator(train_loader)

    logger.info('Building model...')
    model = TranMResProModel(
        config.model,
        protein_res_feature_dim=(27, 3),
        ligand_atom_feature_dim=(13, 1),
        num_props=config.train.num_props,
    ).to(args.device)
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    ckpt_path = ("/workspace/kangchenglong/drug_design_protein/logs/train_res_2025_03_06__13_10_17/checkpoints/model_epoch4.pt")
    model, optimizer, scheduler, start_epoch, best_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
    early_stopping = EarlyStopping('min', 20, delta=0.00005)

    # retrival
    with open(r'data/retrival_database/retrieval_topsim100_chembl_92.pkl', 'rb') as f:
        train_retrieval = pickle.load(f)

    # Precompute and cache MolT5 embeddings
    molT5_embedding_module = MolT5EmbeddingModule(model_path="models/molT5").to(args.device)
    molT5_embedding_cache = {}
    with torch.no_grad():
        for smile, retrieval_list in tqdm(train_retrieval.items()):
            molT5_embedding_cache[smile] = retrieval_list[:args.k]


    def train(verbose=1, num_epoches=200):
        train_losses = []
        val_losses = []
        start_epoch = 5
        best_loss = 1000
        logger.info('start training...')

        for epoch in tqdm(range(num_epoches)):
            model.train()
            batch_losses = []
            batch_cnt = 0

            # for each epoch should update retrival database

            for batch in train_loader:
                batch = batch.to(args.device)
                smile_list = batch['ligand_smile']

                # retrival data
                batch_retricval_list = itemgetter(*smile_list)(molT5_embedding_cache)
                with torch.no_grad():
                    ret_embeddings = molT5_embedding_module(batch_retricval_list, args.device)
                batch_ret_embeddings = ret_embeddings.reshape(len(smile_list), args.k, -1)
                batch_cnt += 1

                dic = {'sas': batch.ligand_sas,
                       'logP': batch.ligand_logP,
                       'qed': batch.ligand_qed,
                       'tpsa': batch.ligand_tpsa,
                       'vina_score': batch.vina_score,
                       }
                if config.train.num_props:
                    dic['vina_score'] = (torch.lt(dic['vina_score'], -7.5)).float()
                    dic['qed'] = (torch.gt(dic['qed'], 0.6)).float()
                    dic['sas'] = (torch.lt(dic['sas'], 4.0)).float()
                    props = config.train.prop
                    prop = torch.tensor(list(zip(*[dic[p] for p in props]))).to(args.device)
                else:
                    prop = None

                outputs = model(
                    batch=batch['batch'],
                    smiles_index=batch['ligand_smiIndices_input'],
                    compose_feature=batch['compose_feature'].float(),
                    compose_vec=batch['compose_vec'],
                    idx_ligand=batch['idx_ligand_ctx_in_compose'],
                    idx_protein=batch['idx_protein_in_compose'],
                    compose_pos=batch['compose_pos'],
                    compose_knn_edge_index=batch['compose_knn_edge_index'],
                    compose_knn_edge_feature=batch['compose_knn_edge_feature'],
                    tgt_len=200,
                    ret_embeddings=batch_ret_embeddings,
                    prop=prop,
                    device=args.device
                )

                loss = criterion(outputs, batch['ligand_smiIndices_tgt'].contiguous().view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                logger.info(
                    'Training Epoch %d | Step %d | Loss %.6f  ' % (
                        epoch + start_epoch, batch_cnt, loss.item()
                    ))

            average_loss = sum(batch_losses) / (len(batch_losses) + 1)
            train_losses.append(average_loss)


            if verbose:
                logger.info(
                    'Training Epoch %d | Average_Loss %.5f ' % (
                        epoch + start_epoch, average_loss
                    ))

            scheduler.step(average_loss)

            if epoch==0 or (len(train_losses) > 1 and train_losses[-1] < train_losses[-2]):
                if config.train.save:
                    ckpt_path = os.path.join(ckpt_dir, 'model_epoch%d.pt' % (epoch + start_epoch))
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': start_epoch + epoch,
                        'best_loss': best_loss
                    }, ckpt_path)

            torch.cuda.empty_cache()


    try:
        train()
    except Exception as e:
        traceback.print_exc()
