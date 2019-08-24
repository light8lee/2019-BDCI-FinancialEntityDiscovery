import numpy as np
import os
from proj_utils.files import save_ckpt, load_ckpt, load_config_from_json
from scipy.stats import pearsonr
from collections import Counter

Precision = lambda tp, fp: tp / (tp + fp) if (tp + fp) != 0 else 0
Recall = lambda tp, fn: tp / (tp + fn) if (tp + fn) != 0 else 0
F1 = lambda p, r: ((2 * p * r) / (p + r)) if (p != 0) and (r != 0) else 0

def sts_metric_builder(args, scheduler_config, model, optimizer, scheduler, writer, Log):
    best_pea = 0
    epoch_loss = 10000
    running_preds = []
    running_labels = []
    running_loss = 0.
    running_size = 0
    def pre_fn():
        nonlocal running_preds
        nonlocal running_labels
        nonlocal running_loss
        nonlocal running_size

        running_preds = []
        running_labels = []
        running_loss = 0.
        running_size = 0

    def step_fn(result, loss, pbar, phase):
        nonlocal running_preds
        nonlocal running_labels
        nonlocal running_loss
        nonlocal running_size

        running_preds.append(result['preds'])
        running_labels.append(result['labels'])
        running_loss += loss.item()
        running_size += result['size']
        if phase == 'train':
            curr_lr = optimizer.param_groups[0]['lr']
            if args.multi_gpu:
                pbar.set_postfix(rank=args.local_rank, mean_loss=running_loss/running_size, lr=curr_lr)
            else:
                pbar.set_postfix(mean_loss=running_loss/running_size, lr=curr_lr)
        else:
            pbar.set_postfix(mean_loss=running_loss/running_size)

    def post_fn(phase, epoch):
        nonlocal running_preds
        nonlocal running_labels
        nonlocal running_loss
        nonlocal running_size
        nonlocal best_pea
        nonlocal epoch_loss

        epoch_loss = running_loss / running_size
        preds = np.concatenate(running_preds, axis=0)
        labels = np.concatenate(running_labels, axis=0)
        epoch_pea = pearsonr(preds, labels)[0][0]
        if args.log:
            writer.add_scalars('loss', {
                phase: epoch_loss,
            }, epoch)
        if phase == 'dev':
            if scheduler_config.name == 'ReduceLROnPlateau':
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
            if epoch_pea > best_pea:
                best_pea = epoch_pea
                if args.multi_gpu:
                    if args.local_rank == 0:
                        Log('dev Epoch {}: Saving Rank({}) New Record... Pea: {}, Loss: {}'.format(
                            epoch, args.local_rank, epoch_pea, epoch_loss))
                        # save_ckpt(os.path.join(args.save_dir, 'model{}.rank{}.epoch{}.pt.tar'.format(args.fold, args.local_rank, epoch)),
                        #                     epoch, model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                        save_ckpt(os.path.join(args.save_dir, 'model{}.rank{}.best.pt.tar'.format(args.fold, args.local_rank)),
                                            epoch, model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                else:
                    Log('dev Epoch {}: Saving New Record... Pea: {}, Loss: {}'.format(
                        epoch, epoch_pea, epoch_loss))
                    # save_ckpt(os.path.join(args.save_dir, 'model{}.epoch{}.pt.tar'.format(args.fold, epoch)),
                    #                        epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                    save_ckpt(os.path.join(args.save_dir, 'model{}.best.pt.tar'.format(args.fold)),
                                            epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
            else:
                if args.multi_gpu:
                    Log('dev Epoch {}: Rank({}) Not Improved. Pea: {}, Loss: {}'.format(
                        epoch, args.local_rank, epoch_pea, epoch_loss))
                else:
                    Log('dev Epoch {}: Not Improved. Pea: {}, Loss: {}'.format(
                        epoch, epoch_pea, epoch_loss))
        else:
            Log('{} Epoch {}: Pea: {}, Loss: {}'.format(
                phase, epoch, epoch_pea, epoch_loss))
    return pre_fn, step_fn, post_fn


def qqp_metric_builder(args, scheduler_config, model, optimizer, scheduler, writer, Log):
    best_f1 = 0
    epoch_loss = 10000
    running_loss = 0.
    running_results = Counter()
    def pre_fn():
        nonlocal running_loss
        nonlocal running_results

        running_loss = 0.
        running_results = Counter()

    def step_fn(result, loss, pbar, phase):
        nonlocal running_loss
        nonlocal running_results

        running_loss += loss.item()
        running_results += result
        if phase == 'train':
            curr_lr = optimizer.param_groups[0]['lr']
            if args.multi_gpu:
                pbar.set_postfix(rank=args.local_rank, mean_loss=running_loss/running_results['total'], mean_acc=(running_results['tp']+running_results['tn'])/running_results['total'], lr=curr_lr)
            else:
                pbar.set_postfix(mean_loss=running_loss/running_results['total'], mean_acc=(running_results['tp']+running_results['tn'])/running_results['total'], lr=curr_lr)
        else:
            pbar.set_postfix(mean_loss=running_loss/running_results['total'], mean_acc=(running_results['tp']+running_results['tn'])/running_results['total'])

    def post_fn(phase, epoch):
        nonlocal epoch_loss
        nonlocal running_loss
        nonlocal running_results
        nonlocal best_f1

        epoch_loss = running_loss / running_results['total']
        epoch_precision = Precision(running_results['tp'], running_results['fp'])
        epoch_recall = Recall(running_results['tp'], running_results['fn'])
        epoch_acc = (running_results['tp'] + running_results['tn']) / running_results['total']
        epoch_f1 = F1(epoch_precision, epoch_recall)
        if args.log:
            writer.add_scalars('loss', {
                phase: epoch_loss,
            }, epoch)
            writer.add_scalars('acc', {
                phase: epoch_acc,
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
                        Log('dev Epoch {}: Saving Rank({}) New Record... Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                            epoch, args.local_rank, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                        # save_ckpt(os.path.join(args.save_dir, 'model{}.rank{}.epoch{}.pt.tar'.format(args.fold, args.local_rank, epoch)),
                        #                     epoch, model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                        save_ckpt(os.path.join(args.save_dir, 'model{}.rank{}.best.pt.tar'.format(args.fold, args.local_rank)),
                                            epoch, model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                else:
                    Log('dev Epoch {}: Saving New Record... Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                        epoch, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                    # save_ckpt(os.path.join(args.save_dir, 'model{}.epoch{}.pt.tar'.format(args.fold, epoch)),
                    #                        epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                    save_ckpt(os.path.join(args.save_dir, 'model{}.best.pt.tar'.format(args.fold)),
                                            epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
            else:
                if args.multi_gpu:
                    Log('dev Epoch {}:  Rank({}) Not Improved. Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                        epoch, args.local_rank, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                else:
                    Log('dev Epoch {}: Not Improved. Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                        epoch, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
        else:
            Log('{} Epoch {}: Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                phase, epoch, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
    return pre_fn, step_fn, post_fn