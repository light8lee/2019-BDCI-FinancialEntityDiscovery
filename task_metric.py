import numpy as np
import os
from proj_utils.files import save_ckpt, load_ckpt, load_config_from_json
from scipy.stats import pearsonr
from collections import Counter
from proj_utils import BIO_TAG2ID


def Precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0


def Recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) != 0 else 0


def F1(p, r):
    return ((2 * p * r) / (p + r)) if (p != 0) and (r != 0) else 0


BIO_BEGIN_TAG_ID = BIO_TAG2ID['B']
BIO_INTER_TAG_ID = BIO_TAG2ID['I']
OTHER_TAG_IDS = [BIO_TAG2ID[v] for v in ['O', '[CLS]', '[SEP]']]
Invalid_entities = set()


def get_BO_entities(batch_tag_ids, max_lens):
    max_lens = [int(v) for v in max_lens]
    for tag_ids, max_len in zip(batch_tag_ids, max_lens):
        entities = set()
        status = 0
        for idx in range(1, max_len):  # not consider [CLS] and [SEP]
            tag_id = tag_ids[idx]
            if (status == 0) and (tag_id == BIO_BEGIN_TAG_ID):  # correct begin
                status = 1
                begin_pos = idx
            elif (status == 1) and (tag_id != BIO_BEGIN_TAG_ID):  # Bx -> O
                status = 0
                entities.add((begin_pos, idx))
        yield entities


def get_BIO_entities(batch_tag_ids, max_lens):
    max_lens = [int(v) for v in max_lens]
    for tag_ids, max_len in zip(batch_tag_ids, max_lens):
        entities = set()
        status = 0
        for idx in range(1, max_len):  # not consider [CLS] and [SEP]
            tag_id = tag_ids[idx]
            if (status == 0) and (tag_id == BIO_BEGIN_TAG_ID):  # correct begin
                status = 1
                begin_pos = idx
            elif (status == 1) and (tag_id in OTHER_TAG_IDS):  # Bx(Ix) -> O
                status = 0
                entities.add((begin_pos, idx))
            elif (status == 1) and (tag_id == BIO_BEGIN_TAG_ID):
                status = 1
                entities.add((begin_pos, idx))
                begin_pos = idx
        yield entities


def get_BIO_entities_v2(batch_tag_ids_list, max_lens, min_count, batch_inputs):
    max_lens = [int(v) for v in max_lens]
    # for tag_ids, max_len in zip(batch_tag_ids, max_lens):
    for i, max_len in enumerate(max_lens):
        inputs = batch_inputs[i]
        dun_hao = inputs.count('、')
        status = 0
        tag_ids = []
        for j in range(max_len):
            counter = Counter()
            print(inputs[j], end=' ')
            for batch_tag_ids in batch_tag_ids_list:
                char = batch_tag_ids[i][j]
                print(char, end=' ')
                counter[char] += 1
            # value = counter.most_common(1)[0]
            # values = [v for v in counter.most_common() if v[1] >= min_count]  # (tag, count)
            # values.sort(key=lambda v: (v[1], -v[0]), reverse=True)  # 先按照出现次数降序，然后按照OBI升序
            if status == 0 and dun_hao > 5 and inputs[j-1] == '、' and '、' in inputs[j+1:j+12] and tag_ids[j-2] == BIO_INTER_TAG_ID:
                status = 1
                tag_ids.append(BIO_BEGIN_TAG_ID)
            elif status == 1 and inputs[j] != '、':
                tag_ids.append(BIO_INTER_TAG_ID)
            elif counter[BIO_BEGIN_TAG_ID] >= min_count:
                tag_ids.append(BIO_BEGIN_TAG_ID)
                status = 0
            elif counter[BIO_INTER_TAG_ID] >= min_count:
                tag_ids.append(BIO_INTER_TAG_ID)
                status = 0
            else:
                status = 0
                tag_ids.append(0)
            print('=>', tag_ids[-1])

        # print(tag_ids)
        entities = set()
        status = 0
        for idx in range(1, max_len):  # not consider [CLS] and [SEP]
            tag_id = tag_ids[idx]
            if (status == 0) and (tag_id == BIO_BEGIN_TAG_ID):  # correct begin
                status = 1
                begin_pos = idx
            elif (status == 1) and (tag_id in OTHER_TAG_IDS):  # Bx(Ix) -> O
                status = 0
                entities.add((begin_pos, idx))
            elif (status == 1) and (tag_id == BIO_BEGIN_TAG_ID):
                status = 1
                entities.add((begin_pos, idx))
                begin_pos = idx
        yield entities


def acc_metric_builder(args, scheduler_config, model, optimizer, scheduler, writer, Log):
    best_f1 = 0
    epoch_loss = 10000
    running_loss = 0.
    running_tp = 0
    running_fp = 0
    running_fn = 0
    running_size = 0
    steps = 0
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
        nonlocal steps
        global Invalid_entities
        steps += 1

        running_size += result['batch_size']
        if phase == 'dev':
            if args.tag_type == 'BIO':
                pred_gen = get_BIO_entities(result['pred_tag_ids'], result['max_lens'])
                target_gen = get_BIO_entities(result['target_tag_ids'], result['max_lens'])
            elif args.tag_type == 'BO':
                pred_gen = get_BO_entities(result['pred_tag_ids'], result['max_lens'])
                target_gen = get_BO_entities(result['target_tag_ids'], result['max_lens'])
            for target_entities, pred_entities, inputs in zip(target_gen, pred_gen, result['inputs']):
                Invalid_entities.update(''.join(inputs[e[0]:e[1]]) for e in (pred_entities-target_entities))
                target_entity_set = set()
                Log('inputs:', ''.join(inputs))
                for start, end in target_entities:
                    target_entity_set.add(''.join(inputs[start:end]))
                pred_entity_set = set()
                for start, end in pred_entities:
                    pred_entity_set.add(''.join(inputs[start:end]))
                Log('target:', target_entity_set)
                Log('predict:', pred_entity_set)
                running_fn += len(target_entity_set-pred_entity_set)
                running_fp += len(pred_entity_set-target_entity_set)
                running_tp += len(pred_entity_set&target_entity_set)
        if phase == 'train':
            running_loss += loss.item()
            curr_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(mean_loss=running_loss/running_size,lr=curr_lr)
            if args.save_steps > 0 and steps % args.save_steps == 0:
                Log('Saving steps:', steps)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                save_ckpt(os.path.join(args.save_dir, 'model{}.step{}.pt.tar'.format(args.fold, steps)),
                                        '', model_to_save.state_dict(), optimizer.state_dict(), scheduler.state_dict())
        else:
            pbar.set_postfix()

    def post_fn(phase, epoch):
        nonlocal epoch_loss
        nonlocal running_loss
        nonlocal best_f1
        Log('epoch:', epoch)

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
                Log('dev Epoch {}: Saving New Record... P: {}, R: {}, F1: {} Loss: {}'.format(
                    epoch, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                save_ckpt(os.path.join(args.save_dir, 'model{}.best.pt.tar'.format(args.fold)),
                                        epoch, model_to_save.state_dict(), optimizer.state_dict(), scheduler.state_dict())
            else:
                Log('dev Epoch {}: Not Improved.  P: {}, R: {}, F1: {} Loss: {}'.format(
                    epoch, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
        else:
            # Log('{} Epoch {}: P: {}, R: {}, F1: {} Loss: {}'.format(
            #     phase, epoch, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            save_ckpt(os.path.join(args.save_dir, 'model{}.epoch{}.pt.tar'.format(args.fold, epoch)),
                                    epoch, model_to_save.state_dict(), optimizer.state_dict(), scheduler.state_dict())
    return pre_fn, step_fn, post_fn


def mrc_acc_metric_builder(args, scheduler_config, model, optimizer, scheduler, writer, Log):
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

        running_size += result['batch_size']
        if phase == 'dev':
            for target_entities, pred_entities, inputs, max_len in zip(result['target_pairs'], result['predict_pairs'], result['inputs'], result['max_lens']):
                target_entities = set(target_entities)
                pred_entities = set([entity for entity in pred_entities if entity[0] < max_len and entity[1] < max_len])
                Invalid_entities.update(''.join(inputs[e[0]:e[1]+1]) for e in (pred_entities-target_entities))
                target_entity_set = set()
                Log('inputs:', ''.join(inputs))
                for start, end in target_entities:
                    target_entity_set.add(''.join(inputs[start:end+1]))
                pred_entity_set = set()
                for start, end in pred_entities:
                    pred_entity_set.add(''.join(inputs[start:end+1]))
                Log('target:', target_entity_set)
                Log('predict:', pred_entity_set)
                running_fn += len(target_entity_set-pred_entity_set)
                running_fp += len(pred_entity_set-target_entity_set)
                running_tp += len(pred_entity_set&target_entity_set)
        if phase == 'train':
            running_loss += loss.item()
            curr_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(mean_loss=running_loss/running_size,lr=curr_lr)
        else:
            pbar.set_postfix()

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
                Log('dev Epoch {}: Saving New Record... P: {}, R: {}, F1: {} Loss: {}'.format(
                    epoch, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                save_ckpt(os.path.join(args.save_dir, 'model{}.best.pt.tar'.format(args.fold)),
                                        epoch, model_to_save.state_dict(), optimizer.state_dict(), scheduler.state_dict())
            else:
                Log('dev Epoch {}: Not Improved.  P: {}, R: {}, F1: {} Loss: {}'.format(
                    epoch, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
        else:
            # Log('{} Epoch {}: P: {}, R: {}, F1: {} Loss: {}'.format(
            #     phase, epoch, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            save_ckpt(os.path.join(args.save_dir, 'model{}.epoch{}.pt.tar'.format(args.fold, epoch)),
                                    epoch, model_to_save.state_dict(), optimizer.state_dict(), scheduler.state_dict())
    return pre_fn, step_fn, post_fn