import os
import time
import argparse
import datetime
import torch.backends.cudnn as cudnn
from timm.utils import accuracy, AverageMeter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from config import get_config
from models import build_model
from data.build import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
import torch
import torch.nn as nn
import numpy as np
from logger import Logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
import nni
from torchviz import make_dot

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

amp = None
def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy().reshape(-1)
    hazards = hazards.cpu().numpy().reshape(-1)
    label = []
    hazard = []
    surv_time = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])

    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)

    return (concordance_index(new_surv, -new_hazard, new_label))

def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)

def _neg_partial_log(prediction, T, E):
    """
    calculate cox loss, Pytorch implementation by Huang, https://github.com/huangzhii/SALMON
    :param X: variables
    :param T: Time
    :param E: Status
    :return: neg log of the likelihood
    """

    current_batch_len = len(prediction)
    R_matrix_train = np.zeros((current_batch_len, current_batch_len), dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()
    # train_ystatus = torch.FloatTensor(E)
    train_ystatus = E.float()
    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)
    return loss_nn

def parse_option(nni_config):
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    args, unparsed = parser.parse_known_args()

    config = get_config(args, nni_config)

    return args, config


def main(config):
    logger = Logger('Big_Brother')
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader(config)
    print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    print(str(model))
    optimizer = build_optimizer(config, model)
    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    # model = nn.DataParallel(model, device_ids=[0, 1, 3])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  #统计需要更新的参数
    print(f"number of params: {n_parameters}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    max_cindex = 0.0
    if config.MODEL.RESUME:
        print('ayyyy!!!!')
        max_cindex = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)
        validate_once(data_loader_test, model)
        print("test_model")
    print("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler)
        loss, c_index, pvalue = validate(data_loader_val, model)
        nni.report_intermediate_result(c_index)
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1) or c_index >= max_cindex:
            print('skip save')
            # model_without_ddp = model.module
            # save_checkpoint(config, epoch, model_without_ddp, c_index, optimizer, lr_scheduler, logger, loss)
        print(f"c_index of the network on the {len(dataset_val)} test images: {c_index:.3f}")
        print(f"pvalue of the network on the {len(dataset_val)} test images: {pvalue:.3f}")
        print(f"loss of the network on the {len(dataset_val)} test images: {loss:.3f}")
        max_cindex = max(max_cindex, c_index)
        print(f'Max c_index: {max_cindex:.3f}')
    loss, c_index, pvalue = validate(data_loader_test, model)
    nni.report_final_result(c_index)
    print(f"c_index of the network on the {len(dataset_test)} test images: {c_index:.3f}")
    print(f"pvalue of the network on the {len(dataset_test)} test images: {pvalue:.3f}")
    print(f"loss of the network on the {len(dataset_test)} test images: {loss:.3f}")
    print(f'test c_index: {c_index:.3f}')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_nn_all = []
    start = time.time()
    end = time.time()
    for idx, (samples, images, targets) in enumerate(data_loader):
        samples = samples.type(torch.FloatTensor)
        nuclei_mask = samples[:, :, :, 0]
        samples = samples[:, :, :, 1:]
        nuclei_mask = nuclei_mask.unsqueeze(3)
        nuclei_mask = nuclei_mask.cuda()
        samples = samples.cuda()
        images = images.cuda()
        lbl_pred = model(samples, images, nuclei_mask)
        # dot = make_dot(lbl_pred)
        # print(dot)
        # dot.format = 'svg'
        # dot.render('autograph')
        cencor_label = targets[:, 0]  # 0 or whatever: cencored  1:monitored
        cencor_label = np.where(cencor_label != 1, 0, 1)
        cencor_label = torch.from_numpy(cencor_label).cuda()
        survtimes_label = targets[:, 1]  # time calculated by day
        survtimes_label.cuda()
        loss_surv = _neg_partial_log(lbl_pred, survtimes_label, cencor_label)
        l1_reg = None
        for W in model.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
        loss = loss_surv + 1e-5 * l1_reg
        b = torch.tensor(config.TRAIN.B)
        loss = (loss-b).abs()+b
        optimizer.zero_grad()
        if amp is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.empty_cache()
        loss_nn_all.append(loss.data.item())

        # if idx == 0:
        #     lbl_pred_all = lbl_pred
        #     survtime_torch = targets[1]
        #     lbl_torch = targets[0]
        # if iter == 0:
        #     lbl_pred_each = lbl_pred
        # else:
        #     lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
        #     lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred])
        #     lbl_torch = torch.cat([lbl_torch, cencor_label])
        #     survtime_torch = torch.cat([survtime_torch, survtimes_label])
        #
        # iter += 1
        # if iter % 16 == 0 or idx == len(data_loader) -1:
        #     survtime_all = np.asarray(survtime_all)
        #     status_all = np.asarray(status_all)
        #     print(status_all)
        #     if np.max(status_all) == 0:
        #         print("encounter no death in a batch, skip")
        #         lbl_pred_each = None
        #         survtime_all = []
        #         status_all = []
        #         iter = 0
        #         continue
        #     optimizer.zero_grad()
        #     loss_surv = _neg_partial_log(lbl_pred_each, survtime_all, status_all)
        #     l1_reg = None
        #     for W in model.parameters():
        #         if l1_reg is None:
        #             l1_reg = torch.abs(W).sum()
        #     else:
        #         l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
        #     loss = loss_surv + 1e-5 * l1_reg
        #     loss.backward()
        #     optimizer.step()
        #     lr_scheduler.step_update(epoch * num_steps + idx)
        #     torch.cuda.empty_cache()
        #     lbl_pred_each = None
        #     survtime_all = []
        #     status_all = []
        #     loss_nn_all.append(loss.data.item())
        #     iter = 0
        #
        # torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            print(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f}\t'
                f'loss {loss:.4f}\t'
                f'grad_norm {norm_meter.avg:.4f}\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate_once(data_loader, model):
    model.eval()
    for idx, (samples, images, targets) in enumerate(data_loader):
        samples = samples.type(torch.FloatTensor)
        nuclei_mask = samples[:, :, :, 0]
        samples = samples[:, :, :, 1:]
        nuclei_mask = nuclei_mask.unsqueeze(3)
        nuclei_mask = nuclei_mask.cuda()
        samples = samples.cuda()
        B, H, W, C = samples.shape
        images = images.cuda()
        for index in range(B):
            print(index)
            temp_sample = samples[index, :, :, :].unsqueeze(0)
            temp_image = images[index, :, :, :].unsqueeze(0)
            temp_nuclei_mask = nuclei_mask[index, :, :, :].unsqueeze(0)
            lbl_pred = model(temp_sample, temp_image, temp_nuclei_mask)
        torch.save(targets, 'label.pt')
        cencor_label = targets[:, 0]
        cencor_label = np.where(cencor_label != 1, 0, 1)
        cencor_label = torch.from_numpy(cencor_label).cuda()
        survtimes_label = targets[:, 1]
        survtimes_label.cuda()


@torch.no_grad()
def validate(data_loader, model):
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cindex_meter = AverageMeter()
    pvalue_meter = AverageMeter()
    end = time.time()
    for idx, (samples, images, targets) in enumerate(data_loader):
        samples = samples.type(torch.FloatTensor)
        nuclei_mask = samples[:, :, :, 0]
        samples = samples[:, :, :, 1:]
        nuclei_mask = nuclei_mask.unsqueeze(3)
        nuclei_mask = nuclei_mask.cuda()
        samples = samples.cuda()
        B, H, W, C = samples.shape
        images = images.cuda()
        lbl_pred = model(samples, images, nuclei_mask)
        cencor_label = targets[:, 0]
        cencor_label = np.where(cencor_label != 1, 0, 1)
        cencor_label = torch.from_numpy(cencor_label).cuda()
        survtimes_label = targets[:, 1]
        survtimes_label.cuda()
        loss_surv = _neg_partial_log(lbl_pred, survtimes_label, cencor_label)
        l1_reg = None
        for W in model.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
        loss = loss_surv + 1e-5 * l1_reg
        pvalue_pred = cox_log_rank(lbl_pred.data, cencor_label, survtimes_label)
        c_index = CIndex_lifeline(lbl_pred.data, cencor_label, survtimes_label)
        loss_meter.update(loss)
        cindex_meter.update(c_index)
        pvalue_meter.update(pvalue_pred)
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'C_idex {cindex_meter.val:.3f} ({cindex_meter.avg:.3f})\t'
                f'pvalue_pred {pvalue_meter.val:.3f} ({pvalue_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    print('\n[val]\t loss (nn):{:.4f}'.format(loss_meter.avg),
                  'c_index: {:.4f}, p-value: {:.3e}'.format(cindex_meter.avg, pvalue_meter.avg))

    return loss_meter.avg, cindex_meter.avg, pvalue_meter.avg



@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    nni_config = nni.get_next_parameter()
    _, config = parse_option(nni_config)

    # if config.AMP_OPT_LEVEL != "O0":
    #     assert amp is not None, "amp not installed!"
    torch.cuda.set_device("cuda:0")
    # seed = config.SEED
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    cudnn.benchmark = True
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    main(config)