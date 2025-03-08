# -*- coding: utf-8 -*-


# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

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
from clip.clip import load
from segment_anything.modeling.loralib.utils import mark_only_lora_as_trainable, lora_state_dict
from copy import deepcopy

VESSEL_CLASS_NUM = 9

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
parser.add_argument("-task_name", type=str, default="EvoSAM")
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
    "-weight_decay", type=float, default=0.0, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.001, metavar="LR", help="learning rate (absolute lr)"
)

parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:2")
parser.add_argument("--order", type=int, default=0)
# parser.add_argument("--task-num", type=int, default=5)

parser.add_argument("--ridge", type=float, default=1.0)

parser.add_argument("--dataset", type=str, default="vessel_3task", choices=["vessel_3task", "vessel_5task", "prostate"])

######################### LoRA config ##########################
# iencoder_use_lora=False,
# iencoder_lora_config=None,
# decoder_use_lora=False,
# decoder_lora_config=None,
parser.add_argument("--iencoder-use-lora", type=bool, default=True)
parser.add_argument("--iencoder-lora-r", type=int, default=2)
parser.add_argument("--iencoder-lora-alpha", type=float, default=1.0)
parser.add_argument("--iencoder-lora-dropout", type=float, default=0.0)
parser.add_argument('--iencoder_enable_lora', nargs='+', type=bool, default=[True, False, True])

parser.add_argument("--decoder-use-lora", type=bool, default=True)
parser.add_argument("--decoder-lora-r", type=int, default=2)
parser.add_argument("--decoder-lora-alpha", type=float, default=1.0)
parser.add_argument("--decoder-lora-dropout", type=float, default=0.0)
parser.add_argument('--decoder_enable_lora', nargs='+', type=bool, default=[True, False, True])
################################################################

args = parser.parse_args()
if args.dataset == "vessel_3task":
    from datasets.vessel_9cls_3task_datasets_with_bbox_with_dataaug import VesselSeqDatasetWithLabel, VesselTestDataset, VesselSeqDatasetInference
    args.task_num = 3
elif args.dataset == "vessel_5task":
    from datasets.vessel_9cls_5task_datasets_with_bbox_with_dataaug import VesselSeqDatasetWithLabel, VesselTestDataset, VesselSeqDatasetInference
    args.task_num = 5
elif args.dataset == "prostate":
    from datasets.prostate_6task_datasets_with_bbox_with_dataaug import VesselSeqDatasetWithLabel, VesselTestDataset, VesselSeqDatasetInference
    args.task_num = 6

######################### LoRA config ##########################
args.iencoder_lora_config = {
    "r": args.iencoder_lora_r,
    "lora_alpha": args.iencoder_lora_alpha,
    "lora_dropout": args.iencoder_lora_dropout,
    "enable_lora":  args.iencoder_enable_lora,
}

args.decoder_lora_config = {
    "r": args.decoder_lora_r,
    "lora_alpha": args.decoder_lora_alpha,
    "lora_dropout": args.decoder_lora_dropout,
    "enable_lora":  args.decoder_enable_lora,
} 
################################################################

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
        args
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        ##################### moe #####################
        clip_model, preprocess = load("ViT-B/32", device=args.device)
        self.state_dicts = []
        self.vit = clip_model.visual
        self.vit.eval()
        self.vit.to(args.device)

        for param in self.vit.parameters():
            param.requires_grad = False

        self.Q = torch.zeros((512, 0)).to(args.device)
        self.G = torch.zeros((512, 512)).to(args.device)
        self.Wo = None
        self.task_dict = {}
        ###############################################

    ##################################### moe ##########################################
    def train(self, mode: bool = True):
        self.image_encoder.train(mode)
        self.mask_decoder.train(mode)
        self.prompt_encoder.train(mode)
        self.training = mode
        if mode and len(self.state_dicts) > 0:
            self.mask_decoder.load_state_dict(self.state_dicts[-1], strict=False)
        ################################################################################

        

    def forward(self, image, box, train=False, H=1920, W=1920, cropped_image=None):
        
        ################################## moe ###################################
        if not train and len(self.state_dicts) > 0:
            assert not self.training
            self.mask_decoder.train()
            assert image.shape[0] == 1
            feature = self.vit(cropped_image).float()
            logits = feature @ self.Wo.T
            pred = logits[0].argmax().item()
            pred = self.task_dict[pred]
            assert pred < len(self.state_dicts)
            self.mask_decoder.load_state_dict(self.state_dicts[pred], strict=False)
            self.mask_decoder.eval()
        ############################################################################

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
            assert self.training
            ori_res_masks = F.interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            return ori_res_masks
        else:
            assert not self.training
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
    for image, gt2D, boxes, _, cropped_image in tqdm(test_dataloader):
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        cropped_image = cropped_image.to(args.device).half()
        H, W = gt2D.shape[-2:]
        with torch.no_grad():
            medsam_pred = medsam_model(image, boxes_np, train=False, H=H, W=W, cropped_image=cropped_image)
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

    ############################### lora ##############################
    logger.info(f"=> iencoder_use_lora: {args.iencoder_use_lora}")
    logger.info(f"=> iencoder_lora_config: {args.iencoder_lora_config}")
    logger.info(f"=> decoder_use_lora: {args.decoder_use_lora}")
    logger.info(f"=> decoder_lora_config: {args.decoder_lora_config}")
    ################################################################### 

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint,
        ############################ lora ############################
        iencoder_use_lora=args.iencoder_use_lora,
        iencoder_lora_config=args.iencoder_lora_config,
        decoder_use_lora=args.decoder_use_lora,
        decoder_lora_config=args.decoder_lora_config,
        ##############################################################
    )
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        args=args
    ).to(device)
    medsam_model.train()

    ############################## lora ##############################
    if args.iencoder_use_lora:
        mark_only_lora_as_trainable(medsam_model.image_encoder, bias='none')
    if args.decoder_use_lora:
        mark_only_lora_as_trainable(medsam_model.mask_decoder, bias='none')
    ##################################################################

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
    
    eval_all_task(args, medsam_model, None, -1, logger, order=args.order)
    for task in range(args.task_num):
        logger.info(f"=> Start Training task {task}.")

        ############################### lora ##############################
        if task == 0:
            optimizer = torch.optim.Adam(
                img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            for param in medsam_model.image_encoder.parameters():
                param.requires_grad = False

            optimizer = torch.optim.Adam(
                medsam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        ###################################################################

        ############################### lora ##############################
        for name, param in list(medsam_model.image_encoder.named_parameters()) + list(medsam_model.mask_decoder.named_parameters()):
            if param.requires_grad:
                logger.info(f"=> trainable parameters: {name}")
        ###################################################################

        seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        # cross entropy loss
        ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        # %% train
        num_epochs = args.num_epochs
        iter_num = 0
        losses = []
        best_loss = 1e10
        train_dataset = VesselSeqDatasetWithLabel(args.tr_npy_path, order_idx=args.order)
        train_dataset.load_dataset(task)
        
        ########################### moe ###########################
        train_dataset_inference = VesselSeqDatasetInference(args.tr_npy_path, order_idx=args.order)
        train_dataset_inference.load_dataset(task)
        vessel_id = {}
        vessel_labels = train_dataset.get_labels()
        cur_num = medsam_model.Q.shape[1]
        for i, label in enumerate(vessel_labels):
            vessel_id[label] = i
            medsam_model.task_dict[cur_num + i] = task
        logger.info("=> task_dict: " + str(medsam_model.task_dict))
        Q = torch.zeros((512, len(vessel_labels))).to(device)
        
        
        train_dataloader_inference = DataLoader(
            train_dataset_inference,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        ###########################################################
        
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
            
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            for step, (image, gt2D, boxes, _, cropped_image, label) in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                cropped_image = cropped_image.to(device).half()
                # ################################### moe ###################################
                # with torch.no_grad():
                #     features = medsam_model.vit(cropped_image).float()
                # label_id = [vessel_id[l] for l in label]
                # medsam_model.G = medsam_model.G + features.T @ features
                # Q = Q + features.T @ F.one_hot(torch.tensor(label_id).to(device), num_classes=len(vessel_labels)).float()
                # ###########################################################################




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
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                iter_num += 1

                # if iter_num >= 10:  break


            epoch_loss /= step
            losses.append(epoch_loss)

            if epoch == num_epochs - 1:
                for step, (image, gt2D, boxes, _, cropped_image, label) in enumerate(tqdm(train_dataloader_inference)):
                    cropped_image = cropped_image.to(device).half()
                    ################################### moe ###################################
                    with torch.no_grad():
                        features = medsam_model.vit(cropped_image).float()
                    label_id = [vessel_id[l] for l in label]
                    medsam_model.G = medsam_model.G + features.T @ features
                    Q = Q + features.T @ F.one_hot(torch.tensor(label_id).to(device), num_classes=len(vessel_labels)).float()
                    ###########################################################################

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

        ################################## moe ####################################
        medsam_model.Q = torch.cat([medsam_model.Q, Q], dim=1)
        medsam_model.Wo = torch.linalg.solve(
                            medsam_model.G + args.ridge * torch.eye(medsam_model.G.size(dim=0)).to(device), 
                            medsam_model.Q
                          ).T
        medsam_model.state_dicts.append(deepcopy(lora_state_dict(medsam_model.mask_decoder)))
        
        checkpoint_moe = {
                "state_dicts": medsam_model.state_dicts,
                "Q": medsam_model.Q,
                "G": medsam_model.G,
                "Wo": medsam_model.Wo,
                "task_dict": medsam_model.task_dict,
                "task": task,
                "epoch": epoch,
        }
        torch.save(checkpoint_moe, join(model_save_path, f"moe_config_{task}.pth"))
        
        ###########################################################################

        eval_all_task(args, medsam_model, None, task, logger, order=args.order)

if __name__ == "__main__":
    main()
