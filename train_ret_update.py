import json
import os
import re
import shutil
import argparse
import traceback
import torch
from pygments.lexers import sas
from rdkit import Chem
from torch.nn.utils import clip_grad_norm_
from torch.utils import tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from termcolor import colored
from models.search import beam_search
from test_mol.metrics import Estimate_logP, Estimate_QED, Estimate_SA
from utils.docking import QVinaDockingTask
from utils.transforms import *
from utils.train import *
from utils.misc import get_new_log_dir

from tqdm import tqdm
from operator import itemgetter

os.environ["TOKENIZERS_PARALLELISM"] = "False"
from models.TranMResPro import MolT5EmbeddingModule, TranMResProModel
from utils.early_stop import *


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


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def load_checkpoint(ckpt_path, model, optimizer, scheduler):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config = checkpoint['config']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['iteration']
    return model, optimizer, scheduler, start_epoch, config


def main(args):
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = tensorboard.SummaryWriter(log_dir)
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
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, exclude_keys=collate_exclude_keys)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, exclude_keys=collate_exclude_keys)
    print('train_loader process done!')

    train_iterator = inf_iterator(train_loader)

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
    early_stopping = EarlyStopping('min', 20, delta=0.00005)
    start_epoch = 0
    if args.load_checkpoint_dir:
        model, optimizer, scheduler, start_epoch, config = load_checkpoint(args.load_checkpoint_dir, model, optimizer,
                                                                           scheduler)
    # retrival
    with open(r'data/retrival_database/retrieval_topsim100_chembl.pkl', 'rb') as f:
        initial_train_retrieval = pickle.load(f)

    # Precompute and cache MolT5 embeddings
    molT5_embedding_module = MolT5EmbeddingModule(model_path="models/molT5").to(args.device)
    molT5_embedding_cache = {}
    with torch.no_grad():
        for smile, retrieval_list in tqdm(initial_train_retrieval.items()):
            molT5_embedding_cache[smile] = retrieval_list[:args.k]

    #########################################
    def train(it=1, val_step=1):
        model.train()
        # one batch data
        batch = next(train_iterator).to(args.device)
        batch_size = len(batch)

        smile_list = batch['ligand_smile']
        # retrival data
        batch_retricval_list = itemgetter(*smile_list)(molT5_embedding_cache)
        with torch.no_grad():
            ret_embeddings = molT5_embedding_module(batch_retricval_list, args.device)
        batch_ret_embeddings = ret_embeddings.reshape(len(smile_list), args.k, -1)

        # determinate update retrival database # TODO: later consideration
        if val_step < 0:  # args.retrieval_warm_up:
            model.eval()
            # generate molecules
            num_beams = 20
            topk = 10

            if config.train.num_props:
                prop = torch.tensor([config.generate.prop for i in range(batch_size * num_beams)],
                                    dtype=torch.float).to(args.device)
                assert prop.shape[-1] == config.train.num_props
                num = int(bool(config.train.num_props))
            else:
                num = 0
                prop = None
            with torch.no_grad():
                beam_output = model.gen(
                    batch=batch['batch'],
                    compose_feature=batch['compose_feature'].float(),
                    compose_vec=batch['compose_vec'],
                    idx_ligand=batch['idx_ligand_ctx_in_compose'],
                    idx_protein=batch['idx_protein_in_compose'],
                    compose_pos=batch['compose_pos'],
                    compose_knn_edge_index=batch['compose_knn_edge_index'],
                    compose_knn_edge_feature=batch['compose_knn_edge_feature'],
                    ret_embeddings=batch_ret_embeddings,
                    smiVoc=config.model.decoder.smiVoc,
                    num_beams=num_beams,
                    max_length=config.model.decoder.tgt_len + num,
                    prop=prop,
                    topk=topk,
                    device=args.device
                )
            beam_output = beam_output.view(batch_size, topk, -1)

            batch_smi = []
            for i, item in enumerate(beam_output):
                pdb_block = os.path.join(config.dataset.path, batch[i].protein_filename)
                generate = []
                for j in item:
                    smile = [config.model.decoder.smiVoc[n.item()] for n in j.squeeze()]
                    smile = re.sub('[&$^]', '', ''.join(smile))

                    mol = Chem.MolFromSmiles(smile)

                    # if valid,cal prop, else skip
                    if mol is not None:
                        ligand_name = batch[i].protein_filename.split('_pocket10')[0] + '.sdf'
                        lgaind_pdb_block = os.path.join(config.dataset.path, ligand_name)
                        vina_task = QVinaDockingTask(pdb_block, mol, lgaind_pdb_block, use_uff=True)

                        vina_score = vina_task.run_sync()
                        qed = Estimate_QED(smile)
                        sa = Estimate_SA(smile)

                    generate.append(smile)
                batch_smi.append(generate)
            # update smiles list and batch_ret_embeddings

        # set prop
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
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)

        logger.info('[Training] iter %d | Loss %.6f | lr %.6f | grad %.4f ' % (
            it, loss.item(), optimizer.param_groups[0]['lr'], orig_grad_norm))

        return loss.item()

    def validate(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)

                # retrival data
                smile_list = batch['ligand_smile']
                batch_retricval_list = itemgetter(*smile_list)(molT5_embedding_cache)
                with torch.no_grad():
                    ret_embeddings = molT5_embedding_module(batch_retricval_list, args.device)
                batch_ret_embeddings = ret_embeddings.reshape(len(smile_list), args.k, -1)

                # set prop
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
                sum_loss += loss.item()
                sum_n += 1
                del outputs, batch

        avg_loss = sum_loss / sum_n

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(f'[Validate] Iter {it:05d} | Loss {colored(avg_loss, "red")}')

        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        return avg_loss

    def test(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_loader, desc='Test'):
                batch = batch.to(args.device)

                # retrival data
                smile_list = batch['ligand_smile']
                batch_retricval_list = itemgetter(*smile_list)(molT5_embedding_cache)
                with torch.no_grad():
                    ret_embeddings = molT5_embedding_module(batch_retricval_list, args.device)
                batch_ret_embeddings = ret_embeddings.reshape(len(smile_list), args.k, -1)

                # set prop
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
                sum_loss += loss.item()
                sum_n += 1
                del outputs, batch
        avg_loss = sum_loss / sum_n
        logger.info('[Test] Iter %05d | Loss %.6f' % (
            it, avg_loss,
        ))
        writer.add_scalar('Test/loss', avg_loss, it)
        return avg_loss

    def test_gen(it):
        num_beams = 20
        topk = 10
        model.eval()
        with torch.no_grad():
            batch_beam_output = []
            batch_protein_name = []
            for batch in tqdm(test_loader, desc='Test'):
                batch = batch.to(args.device)
                batch_size = len(batch)
                # retrival data
                smile_list = batch['ligand_smile']
                batch_retricval_list = itemgetter(*smile_list)(molT5_embedding_cache)
                with torch.no_grad():
                    ret_embeddings = molT5_embedding_module(batch_retricval_list, args.device)
                batch_ret_embeddings = ret_embeddings.reshape(len(smile_list), args.k, -1)

                # set prop
                if config.train.num_props:
                    prop = torch.tensor([config.generate.prop for i in range(batch_size * num_beams)],
                                        dtype=torch.float).to(args.device)
                    num = int(bool(config.train.num_props))
                else:
                    num = 0
                    prop = None

                beam_output = model.gen(
                    batch=batch['batch'],
                    compose_feature=batch['compose_feature'].float(),
                    compose_vec=batch['compose_vec'],
                    idx_ligand=batch['idx_ligand_ctx_in_compose'],
                    idx_protein=batch['idx_protein_in_compose'],
                    compose_pos=batch['compose_pos'],
                    compose_knn_edge_index=batch['compose_knn_edge_index'],
                    compose_knn_edge_feature=batch['compose_knn_edge_feature'],
                    ret_embeddings=batch_ret_embeddings,
                    smiVoc=config.model.decoder.smiVoc,
                    num_beams=num_beams,
                    max_length=config.model.decoder.tgt_len + num,
                    prop=prop,
                    topk=topk,
                    device=args.device
                )
                beam_output = beam_output.view(batch_size, topk, -1)

                batch_beam_output.append(beam_output)
                batch_protein_name.append(batch.protein_filename)

            generate, sas, qeds, vina_scores = [], [], [], []
            vina_dict = {}
            valid_num = 0
            for batch_i, batch_item in tqdm(enumerate(batch_beam_output), desc='Evaluation'):
                protein_names = batch_protein_name[batch_i]
                for i, item in enumerate(batch_item):
                    pdb_block = os.path.join(config.dataset.path, protein_names[i].split('.')[0])
                    for j in item:
                        smile = [config.model.decoder.smiVoc[n.item()] for n in j.squeeze()]
                        smile = re.sub('[&$^]', '', ''.join(smile))
                        mol = Chem.MolFromSmiles(smile)

                        # if valid,cal prop, else skip
                        if mol is not None:
                            ligand_name = protein_names[i].split('_pocket10')[0] + '.sdf'
                            lgaind_pdb_block = os.path.join(config.dataset.path, ligand_name)
                            vina_task = QVinaDockingTask(pdb_block, mol, lgaind_pdb_block, use_uff=True)
                            try:
                                vina_score = vina_task.run_sync()
                            except:
                                vina_score = 0
                            try:
                                qed = Estimate_QED(smile)
                            except:
                                qed = 0
                            try:
                                sa = Estimate_SA(smile)
                            except:
                                sa = 0

                            valid_num += 1
                        else:
                            vina_score = 0.0
                            qed = 0.0
                            sa = 0.0

                        generate.append(smile)
                        qeds.append(qed)
                        sas.append(sa)
                        vina_scores.append(vina_score)

                        update_dict(vina_dict, protein_names[i], [smile, vina_score, qed, sa])

            info_json = json.dumps(vina_dict, sort_keys=False, indent=4, separators=(',', ': '))

            with open(os.path.join(ckpt_dir, 'gen10_epoch_%d.ckpt' % it), 'w') as f:
                f.write(info_json)
            print('epoch %d the gernerated molecules properties: valid: %.4f vina score: %.4f, qed: %.4f sa: %.4f' % (
                it, valid_num / len(vina_scores), np.mean(vina_scores), np.mean(qeds), np.mean(sas)))

    ###################################################
    logger.info('start training...')
    val_step = 0
    for it in range(start_epoch, config.train.max_iters + 1):
        train(it, val_step)
        if it % config.train.val_freq == 0 or it == config.train.max_iters:
            val_step += 1
            avg_loss = validate(it)
            update, _, counts = early_stopping(avg_loss)

            if update:
                logger.info(colored(f"Loss is decending...", 'red'))
            else:
                logger.info(f"earlystop counter: {counts}/20")

            if early_stopping.early_stop:
                logger.info(f"{'':12s}Encountered earlyStop!")
                logger.info(f"{'':->120s}")

                # save results
                ckpt_path = os.path.join(ckpt_dir, 'model_epoch_%d_early_stop.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
                break

            # save
            ckpt_path = os.path.join(ckpt_dir, 'model_epoch_%d.pt' % it)
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': it,
            }, ckpt_path)
            test_gen(it)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_res.yml')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--k', type=int, help='number of retrival molecules about per input molecule', default=10)
    parser.add_argument('--retrieval_warm_up', type=int, help='number of steps of retrival warm up', default=5)
    parser.add_argument('--batch_size', type=int, help='number of batch size', default=32)
    parser.add_argument('--load_checkpoint_dir', type=str, help='load checkpoint model',
                        default='')

    args = parser.parse_args()
    main(args=args)
