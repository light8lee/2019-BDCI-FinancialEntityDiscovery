import numpy as np
import os
from proj_utils.files import save_ckpt, load_ckpt, load_config_from_json
from scipy.stats import pearsonr
from collections import Counter
from proj_utils import BIO_TAG2ID

Precision = lambda tp, fp: tp / (tp + fp) if (tp + fp) != 0 else 0
Recall = lambda tp, fn: tp / (tp + fn) if (tp + fn) != 0 else 0
F1 = lambda p, r: ((2 * p * r) / (p + r)) if (p != 0) and (r != 0) else 0

BIO_BEGIN_TAG_ID = BIO_TAG2ID['B']
BIO_INTER_TAG_ID = BIO_TAG2ID['I']
OTHER_TAG_IDS = [0, 4]
Invalid_entities = set()

def get_BIO_entities(batch_tag_ids, max_lens):
    max_lens = [int(v) for v in max_lens]
    for tag_ids, max_len in zip(batch_tag_ids, max_lens):
        entities = set()
        status = 0
        for idx in range(1, max_len):  # not consider [CLS] and [SEP]
            tag_id = tag_ids[idx]
            if (status == 0) and (tag_id == BIO_BEGIN_TAG_ID):  # correct begin
                status = 1
                label = tag_id
                begin_pos = idx
            # elif (status == 1) and (tag_id == BIO_INTER_TAG_ID):  # in entity Bx(Ix) -> Ix
            #     continue
            elif (status == 1) and (tag_id in OTHER_TAG_IDS):  # Bx(Ix) -> O
                status = 0
                entities.add((begin_pos, idx))
        yield entities

def acc_metric_builder(args, scheduler_config, model, optimizer, scheduler, writer, Log):
    best_f1 = 0
    epoch_loss = 10000
    running_loss = 0.
    running_tp = 0
    running_fp = 0
    running_fn = 0
    running_size = 0
    def pre_fn():
        nonlocal running_loss
        nonlocal running_fn
        nonlocal running_fp
        nonlocal running_tp
        nonlocal running_size

        running_loss = 0.
        running_tp = 0
        running_fp = 0
        running_fn = 0
        running_size = 0

    def step_fn(result, loss, pbar, phase):
        nonlocal running_loss
        nonlocal running_fn
        nonlocal running_fp
        nonlocal running_tp
        nonlocal running_size
        global Invalid_entities

        running_loss += loss.item()
        running_size += result['batch_size']
        pred_gen = get_BIO_entities(result['pred_tag_ids'], result['max_lens'])
        target_gen = get_BIO_entities(result['target_tag_ids'], result['max_lens'])
        for target_entities, pred_entities, inputs in zip(target_gen, pred_gen, result['inputs']):
            Invalid_entities.update(''.join(inputs[e[0]:e[1]]) for e in (pred_entities-target_entities))
            running_fn += len(target_entities-pred_entities)
            running_fp += len(pred_entities-target_entities)
            running_tp += len(pred_entities&target_entities)
        if phase == 'train':
            curr_lr = optimizer.param_groups[0]['lr']
            if args.multi_gpu:
                pbar.set_postfix(rank=args.local_rank, mean_loss=running_loss/running_size, lr=curr_lr)
            else:
                pbar.set_postfix(mean_loss=running_loss/running_size,lr=curr_lr)
        else:
            pbar.set_postfix(mean_loss=running_loss/running_size)

    def post_fn(phase, epoch):
        nonlocal epoch_loss
        nonlocal running_loss
        nonlocal best_f1

        epoch_loss = running_loss / running_size
        epoch_precision = Precision(running_tp, running_fp)
        epoch_recall = Recall(running_tp, running_fn)
        epoch_f1 = F1(epoch_precision, epoch_recall)
        if args.log:
            writer.add_scalars('loss', {
                phase: epoch_loss,
            }, epoch)
            writer.add_scalars('f1', {
                phase: epoch_f1,
            }, epoch)
        if phase == 'dev':
            if scheduler_config.name == 'ReduceLROnPlateau':
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                if args.multi_gpu:
                    if args.local_rank == 0:
                        Log('dev Epoch {}: Saving Rank({}) New Record... P: {}, R: {}, F1: {} Loss: {}'.format(
                            epoch, args.local_rank, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                        # save_ckpt(os.path.join(args.save_dir, 'model{}.rank{}.epoch{}.pt.tar'.format(args.fold, args.local_rank, epoch)),
                        #                     epoch, model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                        save_ckpt(os.path.join(args.save_dir, 'model{}.rank{}.best.pt.tar'.format(args.fold, args.local_rank)),
                                            epoch, model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                else:
                    Log('dev Epoch {}: Saving New Record... P: {}, R: {}, F1: {} Loss: {}'.format(
                        epoch, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                    # save_ckpt(os.path.join(args.save_dir, 'model{}.epoch{}.pt.tar'.format(args.fold, epoch)),
                    #                        epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                    save_ckpt(os.path.join(args.save_dir, 'model{}.best.pt.tar'.format(args.fold)),
                                            epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
            else:
                if args.multi_gpu:
                    Log('dev Epoch {}:  Rank({}) Not Improved. P: {}, R: {}, F1: {} Loss: {}'.format(
                        epoch, args.local_rank,  epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                else:
                    Log('dev Epoch {}: Not Improved.  P: {}, R: {}, F1: {} Loss: {}'.format(
                        epoch, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
        else:
            Log('{} Epoch {}: P: {}, R: {}, F1: {} Loss: {}'.format(
                phase, epoch, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
            save_ckpt(os.path.join(args.save_dir, 'model{}.epoch{}.pt.tar'.format(args.fold, epoch)),
                                    epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
    return pre_fn, step_fn, post_fn