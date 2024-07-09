import os

import numpy as np
import torch
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss, Tversky_loss, Focal_Tversky_loss,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score



def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Freeze_Epoch, Epoch, cuda, dice_loss, focal_loss, tversky_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, log_dir, alpha, awl, local_rank=0):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        # print('Start Train, lambda = {}'.format(alpha))
        print('Start train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, seg_labels, domain_labels = batch
        pngs = pngs[:pngs.shape[0]//2]
        seg_labels = seg_labels[:seg_labels.shape[0]//2]

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                domain_labels = domain_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, domain_output = model_train(imgs, alpha)
            # ----------------------#
            #   计算损失
            # ----------------------#
            loss = 0.
            if focal_loss:
                seg_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                seg_loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            seg_loss *= 0.8
            loss += seg_loss

            loss_domain = torch.nn.BCELoss()
            loss_d = loss_domain(domain_output, domain_labels)
            loss += loss_d

            if tversky_loss:
                tversky = Focal_Tversky_loss(outputs, seg_labels)
                loss += tversky
                loss_sum = awl(seg_loss, loss_d, tversky)
                # loss_sum = awl(loss_d, tversky)

            if dice_loss:
                main_dice = Dice_loss(outputs, seg_labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, seg_labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            # loss.backward()
            loss_sum.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs, domain_output = model_train(imgs, alpha)
                # ----------------------#
                #   计算损失
                # ----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, seg_labels)
                    loss = loss + main_dice

                if tversky_loss:
                    tversky = Focal_Tversky_loss(outputs, seg_labels)
                    loss = loss + tversky

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, seg_labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # total_loss += loss.item()
        total_loss += loss_sum.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': '{0:1.4f}'.format(total_loss / (iteration + 1)),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer),
                                # 'label': loss.item() - loss_d.item(),
                                # 'domain': loss_d.item()
                                })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, seg_labels, domain_labels = batch
        pngs = pngs[:pngs.shape[0] // 2]
        seg_labels = seg_labels[:seg_labels.shape[0] // 2]
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                domain_labels = domain_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, domain_output = model_train(imgs, alpha)
            # ----------------------#
            #   计算损失
            # ----------------------#
            if focal_loss:
                seg_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                seg_loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            seg_loss *= 0.8
            loss += seg_loss

            loss_domain = torch.nn.BCELoss()
            loss_d = loss_domain(domain_output, domain_labels)
            loss += loss_d

            if tversky_loss:
                tversky = Focal_Tversky_loss(outputs, seg_labels)
                loss += tversky
                loss_sum = awl(seg_loss, loss_d, tversky)
                # loss_sum = awl(loss_d, tversky)
            if dice_loss:
                main_dice = Dice_loss(outputs, seg_labels)
                loss = loss + main_dice
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, seg_labels)

            # val_loss += loss.item()
            val_loss += loss_sum.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': '{0:1.4f}'.format(val_loss / (iteration + 1)),
                                    'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train, alpha)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if epoch > Freeze_Epoch:
            if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
                torch.save(model.state_dict(), os.path.join(log_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

            if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
                print('Save best model to best_epoch_weights.pth')
                torch.save(model.state_dict(), os.path.join(log_dir, "best_epoch_weights.pth"))

            torch.save(model.state_dict(), os.path.join(log_dir, "last_epoch_weights.pth"))





