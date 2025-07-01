import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score, fast_hist, compute_metrics, per_class_PA_Recall, per_class_Precision

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
                  save_dir, local_rank=0, name_classes=None):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    # Initialize confusion matrix
    hist = np.zeros((num_classes, num_classes))

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()

                # Ensure labels_np is in shape (batch_size, height, width)
                if labels_np.ndim == 4:
                    labels_np = np.argmax(labels_np, axis=-1)

                # Flatten the arrays to ensure they are one-dimensional and the same length
                preds = preds.flatten()
                labels_np = labels_np.flatten()

                # Debugging: print shapes
                print(f"Shape of preds: {preds.shape}, Shape of labels_np: {labels_np.shape}")

                hist += fast_hist(labels_np, preds, num_classes)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    labels_np = labels.cpu().numpy()

                    # Ensure labels_np is in shape (batch_size, height, width)
                    if labels_np.ndim == 4:
                        labels_np = np.argmax(labels_np, axis=-1)

                    # Flatten the arrays to ensure they are one-dimensional and the same length
                    preds = preds.flatten()
                    labels_np = labels_np.flatten()

                    # Debugging: print shapes
                    print(f"Shape of preds: {preds.shape}, Shape of labels_np: {labels_np.shape}")

                    hist += fast_hist(labels_np, preds, num_classes)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        avg_train_precision = per_class_Precision(hist)
        avg_train_recall = per_class_PA_Recall(hist)
        print('Finish Train')
        print(f"Average Train Precision: {np.mean(avg_train_precision)}")
        print(f"Average Train Recall: {np.mean(avg_train_recall)}")

        # Save precision and recall to text files
        with open(os.path.join(save_dir, 'train_precision.txt'), 'a') as f:
            f.write(f'Epoch {epoch + 1}: {np.mean(avg_train_precision)}\n')

        with open(os.path.join(save_dir, 'train_recall.txt'), 'a') as f:
            f.write(f'Epoch {epoch + 1}: {np.mean(avg_train_recall)}\n')

        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # Reset confusion matrix for validation
    val_hist = np.zeros((num_classes, num_classes))

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Ensure labels_np is in shape (batch_size, height, width)
            if labels_np.ndim == 4:
                labels_np = np.argmax(labels_np, axis=-1)

            # Flatten the arrays to ensure they are one-dimensional and the same length
            preds = preds.flatten()
            labels_np = labels_np.flatten()

            # Debugging: print shapes
            print(f"Shape of preds: {preds.shape}, Shape of labels_np: {labels_np.shape}")

            val_hist += fast_hist(labels_np, preds, num_classes)

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        avg_val_precision = per_class_Precision(val_hist)
        avg_val_recall = per_class_PA_Recall(val_hist)
        print('Finish Validation')
        print(f"Average Validation Precision: {np.mean(avg_val_precision)}")
        print(f"Average Validation Recall: {np.mean(avg_val_recall)}")
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
            (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

        # Plot and save the confusion matrix
        plot_confusion_matrix(val_hist, classes=name_classes, normalize=True, title='Normalized confusion matrix',
                              save_path=os.path.join(save_dir, 'confusion_matrix.png'))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   损失计算
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(imgs)
                # ----------------------#
                #   损失计算
                # ----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_path:
        plt.savefig(save_path)
    plt.show()

