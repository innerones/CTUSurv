import torch
import numpy as np
import tools
import copy
import torch.nn.functional as F
import torch.nn as nn


def bce_loss(output, target):
    return torch.nn.BCELoss()(torch.sigmoid(output), target.float())
def bce_loss_sum(output, target):
    return torch.nn.BCELoss(reduction='sum')(torch.sigmoid(output), target.float())

def estimate_noise_rate(model, train_loader, val_loader, estimate_loader, optimizer, args, true_tm,
                        filter_outlier=False):
    print('Estimating transition matrix... Please wait...')

    A = torch.zeros((args.nc, args.warmup_epoch, len(train_loader.dataset), 2))
    best_val_loss = float('inf')
    val_loss_list, val_acc_list = [], []
    classwise_val_loss_list = [[] for _ in range(args.nc)]
    classwise_val_acc_list = [[] for _ in range(args.nc)]

    for epoch in range(args.warmup_epoch):
        print(f'Epoch {epoch + 1}')
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda().float(), batch_y.cuda().float()
            batch_y[batch_y == 0] = 1
            batch_y[batch_y == -1] = 0

            optimizer.zero_grad()
            output = model(batch_x)
            loss = bce_loss(output, batch_y)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Train Loss: {total_train_loss / len(train_loader.dataset) * args.bs:.6f}')

        with torch.no_grad():
            model.eval()
            val_loss, val_acc = 0.0, 0
            classwise_val_loss, classwise_val_acc = [0] * args.nc, [0] * args.nc

            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.cuda().float(), batch_y.cuda().float()
                batch_y[batch_y == 0] = 1
                batch_y[batch_y == -1] = 0

                output = model(batch_x)
                loss = bce_loss(output, batch_y)
                val_loss += loss.item()
                predictions = output > 0.5
                val_acc += (predictions == batch_y).sum().item()

                for i in range(args.nc):
                    loss_per_class = bce_loss_sum(output[:, i], batch_y[:, i])
                    classwise_val_loss[i] += loss_per_class.item()
                    classwise_val_acc[i] += (predictions[:, i] == batch_y[:, i]).sum().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())

            print(f'Validation Loss: {val_loss / len(val_loader.dataset) * args.bs:.6f}, '
                  f'Accuracy: {val_acc / (len(val_loader.dataset) * args.nc):.6f}')

            val_loss_list.append(val_loss / len(val_loader.dataset) * args.bs)
            for i in range(args.nc):
                classwise_val_loss_list[i].append(classwise_val_loss[i])
                classwise_val_acc_list[i].append(classwise_val_acc[i])
            val_acc_list.append(val_acc / (len(val_loader.dataset) * args.nc))

        index_num = len(estimate_loader.dataset) // args.bs
        with torch.no_grad():
            model.eval()
            for idx, (batch_x, _) in enumerate(estimate_loader):
                batch_x = batch_x.cuda().float()
                output = torch.sigmoid(model(batch_x)).cpu()

                for i in range(args.nc):
                    prob = torch.stack([1 - output[:, i], output[:, i]]).t()
                    if idx <= index_num:
                        A[i, epoch, idx * args.bs:(idx + 1) * args.bs, :] = prob
                    else:
                        A[i, epoch, index_num * args.bs:len(estimate_loader.dataset), :] = prob

    model.load_state_dict(best_model_state)

    est_T = np.zeros_like(true_tm)
    total_error = 0
    for i in range(args.nc):
        best_epoch = np.argmin(np.array(classwise_val_loss_list[i]))
        print(f'Selected model from epoch {best_epoch}')

        prob_matrix = A[i, best_epoch]
        estimated_T = tools.fit(prob_matrix, 2, filter_outlier)
        estimated_T = tools.norm(estimated_T)

        if true_tm[i, 0, 1] == 0:
            estimated_T[0, 1] = 0
            estimated_T[0, 0] = 1
        if true_tm[i, 1, 0] == 0:
            estimated_T[1, 0] = 0
            estimated_T[1, 1] = 1

        error = tools.error(estimated_T, true_tm[i])
        est_T[i] = estimated_T
        total_error += error
        print(f'Class {i}, Estimated: {estimated_T[[1, 0]]}, True: {true_tm[i, [1, 0]]}, Error: {error}')

    print(f'Total error: {total_error}')
    return est_T
