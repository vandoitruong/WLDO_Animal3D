import sys
sys.path.append("../")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm

from dataset_dogs.base_dataset import AnimalDataset
from model import Model
from global_utils.helpers.visualize import Visualizer
import global_utils.config as config
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='../output', help='Where to export the SMAL fits')
# parser.add_argument('--checkpoint', default='../data/pretrained/model_epoch_00000999.pth', help='Path to network checkpoint')
parser.add_argument('--checkpoint', default='../data/pretrained/3501_00034_betas_v4.pth', help='Path to network checkpoint')
parser.add_argument('--dataset', default='animal3d', choices=['stanford', 'animal_pose'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--num_workers', default=0, type=int, help='Number of processes for data loading')
parser.add_argument('--shape_family_id', default=-1, type=int, help='Shape family to use')
parser.add_argument('--gpu_ids', default="0", type=str, help='GPUs to use. Format as string, e.g. "0,1,2')
parser.add_argument('--param_dir', default="NONE", type=str, help='Exported parameter folder to load')

# Custom loss functions
def joint_loss(pred_keypoints, gt_keypoints):
    """
    Compute joint reprojection loss using L2 error, considering only visible keypoints.
    
    Parameters:
    pred_keypoints: Tensor of predicted keypoints, shape (batch_size, num_keypoints, 3)
    gt_keypoints: Tensor of ground truth keypoints, shape (batch_size, num_keypoints, 3)
    
    Returns:
    loss_joints: Joint reprojection loss
    """
    # Extract the visibility mask from the ground truth keypoints
    visibility = gt_keypoints[:, :, 2] > 0  # shape (batch_size, num_keypoints)

    # Mask the predicted and ground truth keypoints by visibility
    pred_keypoints_visible = pred_keypoints[:, :, :2][visibility]
    gt_keypoints_visible = gt_keypoints[:, :, :2][visibility].cuda()

    # Compute the L2 distance between visible predicted and ground truth keypoints
    loss_joints = torch.norm(pred_keypoints_visible - gt_keypoints_visible, dim=-1).mean()
    
    return loss_joints

def silhouette_loss(pred_silhouettes, gt_silhouettes):
    return torch.norm(pred_silhouettes - gt_silhouettes.cuda(), dim=-1).mean()

def pose_prior_loss(pred_pose, gt_pose):
    return torch.norm(pred_pose - gt_pose.cuda(), dim=-1).mean()

def shape_prior_loss(pred_shape, gt_shape):
    return torch.norm(pred_shape - gt_shape.cuda(), dim=-1).mean()

def total_loss(preds, gt, stage):
    if stage == 1:
        lambda_joints = 10.0
        lambda_pose = 1.0
        lambda_shape = 1.0
        lambda_sil = 0.0

    elif stage == 2:
        lambda_joints = 10.0
        lambda_pose = 0.5
        lambda_shape = 0.0
        lambda_sil = 100.0


    Ljoints = joint_loss(preds['synth_landmarks'], gt['keypoints'])
    Lsil = silhouette_loss(preds['synth_silhouettes'], gt['seg'])
    Lpose = pose_prior_loss(preds['pose'], gt['pose'])
    Lshape = shape_prior_loss(preds['betas'], gt['betas'])

    total_loss_value = lambda_joints * Ljoints + lambda_sil * Lsil + lambda_pose*Lpose + lambda_shape*Lshape

    return total_loss_value


if __name__ == '__main__':
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(filename='training.log', level=logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print (os.environ['CUDA_VISIBLE_DEVICES'])

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    assert torch.cuda.device_count() == 1, "Currently only 1 GPU is supported"

    # Create new result output directory
    print ("RESULTS: {0}".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        print (f"Unable to find: {args.checkpoint}")
    
    load_from_disk = os.path.exists(args.param_dir)

    model = Model(device, 1, None).to(device)

    train_data = AnimalDataset(
            args.dataset,
            is_train=True, 
            use_augmentation=False)
    
    test_data = AnimalDataset(
            args.dataset,
            is_train=False, 
            use_augmentation=False)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers)
    
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_tqdm_iterator = tqdm(train_loader, desc='Training', total=len(train_loader))
    test_tqdm_iterator = tqdm(test_loader, desc='Testing', total=len(test_loader))

    # Lists to store loss values
    train_losses_stage1 = []
    test_losses_stage1 = []
    train_losses_stage2 = []
    test_losses_stage2 = []

    # Stage 1
    num_epochs_stage1 = 100
    for epoch in range(num_epochs_stage1):
        model.train()
        epoch_train_loss = 0
        for step, batch in enumerate(train_tqdm_iterator):
            preds = model(batch)
            loss = total_loss(preds, batch, stage=1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_loss += loss.item()

        train_losses_stage1.append(epoch_train_loss / len(train_loader))
        logging.info(f"Stage 1: Epoch {epoch + 1}/{num_epochs_stage1}, Training Loss: {train_losses_stage1[-1]}")
        print(f"Stage 1: Epoch {epoch + 1}/{num_epochs_stage1}, Training Loss: {train_losses_stage1[-1]}")
        np.save('train_losses_stage1.npy', np.array(train_losses_stage1))
        # Validating
        
        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(test_tqdm_iterator):
                preds = model(batch)
                loss = total_loss(preds, batch, stage=1)
                epoch_test_loss += loss.item()
        test_losses_stage1.append(epoch_test_loss / len(test_loader))
        logging.info(f"Stage 1: Epoch {epoch + 1}/{num_epochs_stage1}, Testing Loss: {test_losses_stage1[-1]}")
        print(f"Stage 1: Epoch {epoch + 1}/{num_epochs_stage1}, Testing Loss: {test_losses_stage1[-1]}")
        np.save('test_losses_stage1.npy', np.array(test_losses_stage1))

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'{args.output_dir}/model_weights_stage1_{epoch + 1}.pth')

    # Stage 2
    num_epochs_stage2 = 100
    for epoch in range(num_epochs_stage2):
        model.train()
        epoch_train_loss = 0
        for step, batch in enumerate(train_tqdm_iterator):
            preds = model(batch)
            loss = total_loss(preds, batch, stage=2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_loss += loss.item()

        train_losses_stage2.append(epoch_train_loss / len(train_loader))
        logging.info(f"Stage 2: Epoch {epoch + 1}/{num_epochs_stage2}, Training Loss: {train_losses_stage2[-1]}")
        print(f"Stage 2: Epoch {epoch + 1}/{num_epochs_stage1}, Training Loss: {train_losses_stage1[-1]}")
        np.save('train_losses_stage2.npy', np.array(train_losses_stage2))
        # Validating
        
        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(test_tqdm_iterator):
                preds = model(batch)
                loss = total_loss(preds, batch, stage=2)
                epoch_test_loss += loss.item()
        test_losses_stage2.append(epoch_test_loss / len(test_loader))
        np.save('test_losses_stage2.npy', np.array(test_losses_stage2))
        logging.info(f"Stage 2: Epoch {epoch + 1}/{num_epochs_stage2}, Testing Loss: {test_losses_stage2[-1]}")
        print(f"Stage 2: Epoch {epoch + 1}/{num_epochs_stage2}, Testing Loss: {test_losses_stage2[-1]}")
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'{args.output_dir}/model_weights_stage2_{epoch + 1}.pth')

    # Plot and save the loss values
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_stage1, label='Training Loss Stage 1')
    plt.plot(test_losses_stage1, label='Validation Loss Stage 1')
    plt.plot(train_losses_stage2, label='Training Loss Stage 2')
    plt.plot(test_losses_stage2, label='Validation Loss Stage 2')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{args.output_dir}/loss_plot.png')
    plt.show()

    print("------------------COMPLETED TRAINING------------------")