# -*- coding: utf-8 -*-
"""
This code is modified from train_one_gpu.py in https://github.com/bowang-lab/MedSAM.
"""

# %% setup environment
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt


join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import logging
import cv2
import json
from copy import deepcopy
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/json_files",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="SAM-distill")
# parser.add_argument("-task_name", type=str, default="debug")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/checkpoints/sam_vit_b_01ec64.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir/experiments")


# train
parser.add_argument("-num_epochs", type=int, default=50)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=2)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)

parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:2")
parser.add_argument("--order", type=int, default=0)
# parser.add_argument("--task-num", type=int, default=5)
parser.add_argument("--temperture", type=float, default=2.0)
parser.add_argument("--dist-coef", type=float, default=0.00001)

parser.add_argument("--dataset", type=str, default="vessel_3task", choices=["vessel_3task", "vessel_5task", "prostate"])

args = parser.parse_args()
if args.dataset == "vessel_3task":
    from datasets.vessel_9cls_3task_datasets_with_bbox_with_dataaug import VesselSeqDataset, VesselTestDataset
    args.task_num = 3
elif args.dataset == "vessel_5task":
    from datasets.vessel_9cls_5task_datasets_with_bbox_with_dataaug import VesselSeqDataset, VesselTestDataset
    args.task_num = 5
elif args.dataset == "prostate":
    from datasets.prostate_6task_datasets_with_bbox_with_dataaug import VesselSeqDataset, VesselTestDataset
    args.task_num = 6

args.task_name = args.task_name + f"_order{args.order}"
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box, train=False, H=1920, W=1920):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        if train:
            ori_res_masks = F.interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            return ori_res_masks
        else:
            low_res_pred = torch.sigmoid(low_res_masks)  # (1, 1, 256, 256)
            low_res_pred = F.interpolate(
                low_res_pred,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            return low_res_pred

def build_logger(log_path):
    logger = logging.getLogger("logger")
    console = logging.StreamHandler()
    file = logging.FileHandler(log_path, encoding="utf-8")
    fmt = "[%(levelname)s] %(asctime)s: %(message)s"
    console.setFormatter(logging.Formatter(fmt))
    file.setFormatter(logging.Formatter(fmt))
    logger.setLevel(level=logging.INFO)
    logger.addHandler(console)
    logger.addHandler(file)
    return logger

def eval_iou(medsam_model, coda, test_dataloader, task, logger):
    medsam_model.eval()
    ious = []
    for image, gt2D, boxes, _, _ in tqdm(test_dataloader):
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        H, W = gt2D.shape[-2:]
        with torch.no_grad():
            medsam_pred = medsam_model(image, boxes_np, train=False, H=H, W=W)
        medsam_pred = medsam_pred.squeeze().cpu().numpy()
        gt2D = gt2D.squeeze().cpu().numpy()
    
        iou = np.sum((medsam_pred > 0.5) & (gt2D > 0.5)) / np.sum((medsam_pred > 0.5) | (gt2D > 0.5))
        ious.append(iou)
            
    mean_iou = np.mean(ious)
    # logger.info(f"=> Task {task} Mean IoU: {mean_iou}")
    return mean_iou

def eval_all_task(args, medsam_model, coda, task, logger, order=0):
    mean_ious = []
    for i in range(args.task_num):
        test_dataset = VesselTestDataset(args.tr_npy_path, order_idx=order)
        test_dataset.load_dataset(i)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        mean_iou = eval_iou(medsam_model, coda, test_dataloader, task, logger)
        mean_ious.append(np.round(mean_iou, 4))  
    
    logger.info(f"=> Task {task} Mean IoU: {str(mean_ious)}.")

def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    logger = build_logger(join(model_save_path, 'sam_coda' + "_train.log"))
    
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    logger.info(
        "=> Number of total parameters: "+ \
        str(sum(p.numel() for p in medsam_model.parameters()))
    )  # 93735472
    logger.info(
        "=> Number of trainable parameters: " + \
        str(sum(p.numel() for p in medsam_model.parameters() if p.requires_grad))
    )  # 93729252
    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )

    
    logger.info(
        "Number of image encoder and mask decoder parameters: " + \
        str(sum(p.numel() for p in img_mask_encdec_params if p.requires_grad)),
    )  # 93729252
    
    old_model = None

    eval_all_task(args, medsam_model, None, -1, logger, order=args.order)
    # eval_iou(medsam_model, None, test_dataloader, -1, logger)
    for task in range(args.task_num):
        logger.info(f"=> Start Training task {task}.")
    
        optimizer = torch.optim.AdamW(
            img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
        )
        seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        # cross entropy loss
        ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        # %% train
        num_epochs = args.num_epochs
        iter_num = 0
        losses = []
        best_loss = 1e10
        train_dataset = VesselSeqDataset(args.tr_npy_path, order_idx=args.order)
        train_dataset.load_dataset(task)
        logger.info("Number of training samples: " + str(len(train_dataset)))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        medsam_model.train()
        start_epoch = 0
        if args.resume is not None and args.resume != "":
            if os.path.isfile(args.resume):
                raise NotImplementedError
                ## Map model to be loaded to specified single GPU
                checkpoint = torch.load(args.resume, map_location=device)
                start_epoch = checkpoint["epoch"] + 1
                medsam_model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            for step, (image, gt2D, boxes, _, cropped_image) in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)

                #################### vis ####################
                # vis_img = image[0].permute(1,2,0).cpu().numpy()
                # vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
                # vis_img = (vis_img * 255).astype(np.uint8)
                # vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                # vis_img = cv2.rectangle(vis_img, (int(boxes_np[0,0]), int(boxes_np[0,1])), (int(boxes_np[0,2]), int(boxes_np[0,3])), (0,255,0), 2)
                
                # vis_mask = (gt2D[0][0].cpu().numpy() > 0)
                # vis_img[vis_mask, 1] = (255 * 0.2 + vis_img[vis_mask, 1] * 0.8).astype('uint8')
                # cv2.imwrite("vis.jpg", vis_img)

                # crop_vis_img = cropped_image[0].permute(1,2,0).cpu().numpy()
                # crop_vis_img = (crop_vis_img - crop_vis_img.min()) / (crop_vis_img.max() - crop_vis_img.min())
                # crop_vis_img = (crop_vis_img * 255).astype(np.uint8)
                # crop_vis_img = cv2.cvtColor(crop_vis_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite("crop_vis.jpg", crop_vis_img)

                # import pdb; pdb.set_trace()
                #############################################


                if args.use_amp:
                    raise NotImplementedError
                    ## AMP
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        medsam_pred = medsam_model(image, boxes_np, train=True)
                        loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                            medsam_pred, gt2D.float()
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    medsam_pred = medsam_model(image, boxes_np, train=True)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                    
                    if old_model is not None:
                        # import pdb; pdb.set_trace()
                        with torch.no_grad():
                            medsam_pred_old = old_model(image.clone(), boxes_np.copy(), train=True)
                        # distillation with kl_div
                        # medsam_pred_old: B x 1 x 1024 x 1024
                        # medsam_pred: B x 1 x 1024 x 1024
                        T = args.temperture
                        medsam_pred_old = torch.cat([medsam_pred_old, torch.zeros_like(medsam_pred_old)], dim=1)
                        medsam_pred = torch.cat([medsam_pred, torch.zeros_like(medsam_pred)], dim=1)
                        dist_loss = F.kl_div(
                                        F.log_softmax(medsam_pred / T, dim=1), 
                                        F.softmax(medsam_pred_old.detach() / T, dim=1), 
                                        reduction='batchmean'
                                    ) * T * T
                        
                    else:
                        dist_loss = 0.0
                    loss = loss + dist_loss * args.dist_coef
                    
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                iter_num += 1

            epoch_loss /= step
            losses.append(epoch_loss)
            
            logger.info(
                f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
            )
            ## save the latest model
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "task": task,
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, f"medsam_model_{task}.pth"))
            ## save the best model
            # if epoch_loss < best_loss:
            #     best_loss = epoch_loss
            #     checkpoint = {
            #         "model": medsam_model.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "epoch": epoch,
            #     }
            #     torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

            # %% plot loss
        # plt.plot(losses)
        # plt.title("Dice + Cross Entropy Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.savefig(join(model_save_path, args.task_name + f"train_loss_{task}.png"))
        # plt.close()

        eval_all_task(args, medsam_model, None, task, logger, order=args.order)
        # eval_iou(medsam_model, None, test_dataloader, task, logger)
        old_model = deepcopy(medsam_model)
        old_model.eval()

if __name__ == "__main__":
    main()
