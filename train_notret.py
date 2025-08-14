# python train.py --config ./configs/train_res.yml --logdir logs
import os
import shutil
import argparse
import traceback

from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from utils.datasets import *
from utils.transforms import *
from utils.train import *
from models.embedding import GVP,GVPConvLayer,GVPConv,ResEmbedding_GVP
from models.ResGen import ResGen
from utils.datasets.resgen import ResGenDataset
from utils.misc import get_new_log_dir
from utils.train import get_model_loss
from time import time
from models.interaction.cftfm import CFTransformerEncoderVN
from models.TranMResPro import MolT5EmbeddingModule
from models.duibi import DuibiModel
from utils.early_stop import *
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_res.yml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--logdir', type=str, default='./logs')
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
    edge_sampler = EdgeSample(config.train.transform.edgesampler)  # {'k': 8}
    cfg_ctr = config.train.transform.contrastive
    contrastive_sampler = ContrastiveSample(cfg_ctr.num_real, cfg_ctr.num_fake, cfg_ctr.pos_real_std,
                                            cfg_ctr.pos_fake_std,
                                            config.model.field.knn)

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

    collate_exclude_keys = ['ligand_nbh_list']
    import pickle

    # print(train_set[0])
    # print(len(train_set))
    # filtered_train_set_allfe_1000 = [obj for obj in train_set if 'ligand_weight' in obj and 'ligand_logP' in obj and 'ligand_sas' in obj and 'ligand_qed' in obj and 'ligand_tpsa' in obj and 'vina_score' in obj]
    # filtered_val_set_allfe = [obj for obj in val_set if 'ligand_weight' in obj and 'ligand_logP' in obj and 'ligand_sas' in obj and 'ligand_qed' in obj and 'ligand_tpsa' in obj and 'vina_score' in obj]
    collate_exclude_keys = ['ligand_nbh_list']
    # train_loader = DataLoader(train_set,1, shuffle=False, exclude_keys=collate_exclude_keys,drop_last=True)

    # with open('filter_train_set_allfe_1000.pkl', 'wb') as f:
    #     pickle.dump(filtered_train_set_allfe_1000, f)

    with open(r'D:\Transformer-M-main\Transformer-M-main\ResGen-main\filter_train_set_allfe_1000.pkl', 'rb') as f:
        train_set = pickle.load(f)
    print(len(train_set))

    train_loader = DataLoader(train_set, 32, shuffle=False, exclude_keys=collate_exclude_keys, drop_last=True)
    print('train_loader process done!')
    #迭代器
    def inf_iterator(iterable):
        iterator = iterable.__iter__()
        while True:
            try:
                yield iterator.__next__()
            except StopIteration:
                iterator = iterable.__iter__()
    train_interator = inf_iterator(train_loader)


    def load_checkpoint(ckpt_path, model, optimizer, scheduler,device):
        checkpoint = torch.load(ckpt_path,map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        return model, optimizer, scheduler, start_epoch, best_loss
    # Model
    logger.info('Building model...')

    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping('min', 20, delta=0.00005)
    model = DuibiModel(
        config.model,
        protein_res_feature_dim=(27, 3),
        ligand_atom_feature_dim=(13, 1),
        num_props=5,
    ).to(args.device)
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    def train(verbose=1, num_epoches = 50):
        train_losses = []
        val_losses = []
        start_epoch =51
        best_loss = 1000
        logger.info('start training...')

        for epoch in range(num_epoches):
            model.train()
            batch_losses = []
            batch_cnt=0
            for batch in train_loader:
                batch=batch.to(args.device)
                batch_cnt+=1
                # prop = None
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
                    batch=batch.batch,
                    smiles_index=batch.ligand_smiIndices_input,
                    compose_feature=batch.compose_feature.float(),
                    compose_vec=batch.compose_vec,
                    idx_ligand=batch.idx_ligand_ctx_in_compose,
                    idx_protein=batch.idx_protein_in_compose,
                    compose_pos=batch.compose_pos,
                    compose_knn_edge_index=batch.compose_knn_edge_index,
                    compose_knn_edge_feature=batch.compose_knn_edge_feature,
                    tgt_len=200,
                    prop=prop,
                )
                # 将目标张量重新调整为形状 [32, x]
                loss = criterion(outputs, batch.ligand_smiIndices_tgt.contiguous().view(-1))
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
            # if len(train_losses) > 1 :
            # if (train_losses[-1] < train_losses[-2]):
            if config.train.save:
                        ckpt_path = os.path.join(ckpt_dir, 'not_retreival%d.pt' % int(epoch + start_epoch))
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