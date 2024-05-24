import sys
sys.path.append("../")

"""
File for using the model on a single directory
"""

from model import Model

import torch, os
import argparse

from dataset_dogs.demo_dataset import DemoDataset
from torch.utils.data import DataLoader
from global_utils.helpers.visualize import Visualizer

import cv2
import numpy as np
from tqdm import tqdm
import traceback

nn = torch.nn


class DemoModel(nn.Module):
    def __init__( self, device, shape_family_id, load_from_disk, **kwargs):

        super(DemoModel, self).__init__()
        self.module = Model(device, shape_family_id, load_from_disk, **kwargs)

    def forward(self, batch_input, demo=False):
        out = self.module(batch_input, demo)
        return out
ep = 10
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=f'../output/run5/model_weights_stage1_{ep}.pth', help='Path to network checkpoint')
# parser.add_argument('--checkpoint', default='../data/pretrained/3501_00034_betas_v4.pth', help='Path to network checkpoint')
parser.add_argument('--src_dir', default="../example_imgs", type=str, help='The directory of input images')
parser.add_argument('--result_dir', default=f'../demo_out/run5/epoch{ep}', help='Where to export the output data')
parser.add_argument('--shape_family_id', default=1, type=int, help='Shape family to use')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--gpu_ids', default="0", type=str, help='GPUs to use. Format as string, e.g. "0,1,2')

def run_demo(args, device):
    """Run evaluation on the datasets and metrics we report in the paper. """

    os.makedirs(args.result_dir, exist_ok=True)


    if not os.path.exists(args.checkpoint):
        print (f"Unable to find: {args.checkpoint}")
    

    model = load_model_from_disk(
        args.checkpoint, args.shape_family_id, 
        False, device)

    batch_size = args.batch_size

    # Load DataLoader
    dataset = DemoDataset(args.src_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)  # , num_workers=num_workers)

    # Store smal parameters
    smal_pose = np.zeros((len(dataset), 105))
    smal_betas = np.zeros((len(dataset), 26))
    smal_camera = np.zeros((len(dataset), 3))
    smal_joints3d = np.zeros((len(dataset), 26, 3))
    smal_imgname = []
    smal_has_bbox = []

    # Iterate over the entire dataset
    tqdm_iterator = tqdm(data_loader, desc='Eval', total=len(data_loader))
    for step, batch in enumerate(tqdm_iterator):
        with torch.no_grad():
            preds = model(batch, demo=True)

            # make sure we dont overwrite something
            assert not any(k in preds for k in batch.keys())
            preds.update(batch)  # merge everything into one big dict

        curr_batch_size = preds['img'].shape[0]

        smal_pose[step * batch_size:step * batch_size + curr_batch_size] = preds['pose'].data.cpu().numpy()
        smal_betas[step * batch_size:step * batch_size + curr_batch_size] = preds['betas'].data.cpu().numpy()
        smal_camera[step * batch_size:step * batch_size + curr_batch_size] = preds['camera'].data.cpu().numpy()
        smal_joints3d[step * batch_size:step * batch_size + curr_batch_size] = preds['joints_3d'].data.cpu().numpy()

        output_figs = np.transpose(
            Visualizer.generate_demo_output(preds).data.cpu().numpy(),
            (0, 1, 3, 4, 2))

        for img_id in range(len(preds['imgname'])):
            imgname = preds['imgname'][img_id].replace("\\", "/")  # always keep in / format
            output_fig_list = output_figs[img_id]

            path_parts = imgname.split('/')
            smal_imgname.append("{0}/{1}".format(path_parts[-2], path_parts[-1]))
            path_suffix = "{0}_{1}".format(path_parts[-2], path_parts[-1])
            img_file = os.path.join(args.result_dir, path_suffix)
            output_fig = np.hstack(output_fig_list)
            smal_has_bbox.append(preds['has_bbox'][img_id])
            cv2.imwrite(img_file, output_fig[:, :, ::-1] * 255.0)

            npz_file = "{0}.npz".format(os.path.splitext(img_file)[0])
            np.savez_compressed(npz_file,
                                imgname=preds['imgname'][img_id],
                                pose=preds['pose'][img_id].data.cpu().numpy(),
                                betas=preds['betas'][img_id].data.cpu().numpy(),
                                camera=preds['camera'][img_id].data.cpu().numpy(),
                                trans=preds['trans'][img_id].data.cpu().numpy(),
                                has_bbox=preds['has_bbox'][img_id],

                                )

    # Save reconstructions to a file for further processing
    param_file = os.path.join(args.result_dir, 'params.npz')
    np.savez(param_file,
             pose=smal_pose,
             betas=smal_betas,
             camera=smal_camera,
             joints3d=smal_joints3d,
             imgname=smal_imgname,
             has_bbox=smal_has_bbox)

    print("--> Exported param file: {0}".format(param_file))
    print('*** FINISHED ***')

def load_model_from_disk(model_path, shape_family_id, load_from_disk, device):
    model = DemoModel(device, shape_family_id, load_from_disk)
    # model = nn.DataParallel(model)
    model = model.to(device)
    # model.load_state_dict(torch.load(model_path))
    model.eval()
    # model.eval()

    pretrained_dict = torch.load(model_path)

    # model = Model(device, 1, None).to(device)

    model_dict = model.state_dict()

    print("PRETRAIN_DICT")
    # for i, key in enumerate(pretrained_dict.keys()):
    #     parameter = pretrained_dict[key]
    #     # print(f"[{i}]", key, parameter.size())
    #     if i == 2:
    #         print(parameter)
    
    # print("MODEL_DICT")
    # for i, key in enumerate(model_dict.keys()):
    #     parameter = model_dict[key]
    #     # print(f"[{i}]", key, parameter.size())
    #     if i == 2:
    #         print(parameter)

    filtered_pretrained_dict = {}

    # Iterate over each item in pretrained_dict
    for k, v in pretrained_dict.items():
        # Check if the key exists in model_dict
        if k in model_dict:
            # print(k, model_dict[k].shape, pretrained_dict[k].shape)
            if model_dict[k].shape == pretrained_dict[k].shape:
            # If the key exists, add the item to the new dictionary
                filtered_pretrained_dict[k] = v

    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    print("len of fitered dict: ", len(filtered_pretrained_dict))
    # for name, param in filtered_pretrained_dict.items():
    #     try:
    #         model_dict[name].copy_(param)
    #     except Exception as e:
    #         print(f"Unable to load: {name}")
    #         print(f"Error: {e}")
    #         traceback.print_exc()

    model_dict.update(filtered_pretrained_dict)
    model.load_state_dict(model_dict)
    model_dict = model.state_dict()
    

    # print("UPDATED_MODEL_DICT")
    # for i, key in enumerate(model_dict.keys()):
    #     parameter = model_dict[key]
    #     # print(f"[{i}]", key, parameter.size())
    #     if i == 2:
    #         print(parameter)

    # if model_path is not None:
    #     print( "found previous model %s" % model_path )
    #     print( "   -> resuming" )
    #     model_state_dict = torch.load(model_path)

    #     own_state = model.state_dict()
    #     for name, param in model_state_dict.items():
    #         try:
    #             own_state[name].copy_(param)
    #         except Exception as e:
    #             print(f"Unable to load: {name}")
    #             print(f"Error: {e}")
    #             traceback.print_exc()
    # else:
    #     print ('model_path is none')

    print ("LOADED")

    return model


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print (os.environ['CUDA_VISIBLE_DEVICES'])

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    assert torch.cuda.device_count() == 1, "Currently only 1 GPU is supported"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_demo(args, device)
