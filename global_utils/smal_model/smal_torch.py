"""

    PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl 
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from .smal_basics import align_smal_template_to_symmetry_axis #, get_smal_template
from global_utils import config

# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class SMAL(nn.Module):
    def __init__(self, device, shape_family_id=-1, dtype=torch.float):
        super(SMAL, self).__init__()

        # -- Load SMPL params --
        # with open(pkl_path, 'r') as f:
        #     dd = pkl.load(f)
        self.device = device

        print (f"Loading SMAL with shape family: {shape_family_id}")
            
        with open(config.SMAL_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            dd = u.load()

        self.f = dd['f']

        self.faces = torch.from_numpy(self.f.astype(int)).to(device)
        
        # replaced logic in here (which requried SMPL library with L58-L68)
        # v_template = get_smal_template(
        #     model_name=config.SMAL_FILE, 
        #     data_name=config.SMAL_DATA_FILE, 
        #     shape_family_id=shape_family_id)

        v_template = dd['v_template']

        # Size of mesh [Number of vertices, 3]
        self.size = [v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis
        
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T.copy()
        self.shapedirs = Variable(
            torch.Tensor(shapedir), requires_grad=False).to(device)

        with open(config.SMAL_DATA_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()

        # Zero_Betas -> V_Template -> Aligned
        # Zero_Betas -> V_Template -> V_Template + ShapeCluster * ShapeDirs -> Aligned

        # Aligned(V_T + ShapeCluster * ShapeDirs) - ShapeCluster * ShapeDirs

        # Select mean shape for quadruped type
        shape_cluster_means = data['cluster_means'][shape_family_id]
        
        # NOTE: The model was trained using v_template for shape_params=[0]*41
        if shape_family_id != -1:
            v_template = v_template + np.matmul(
                shape_cluster_means[None,:], shapedir).reshape(
                -1, self.size[0], self.size[1])[0]

        self.shape_cluster_means = torch.from_numpy(shape_cluster_means).float().to(device)

        v_sym, self.left_inds, self.right_inds, self.center_inds = align_smal_template_to_symmetry_axis(
            v_template, sym_file=config.SMAL_SYM_FILE)

        # Mean template vertices
        self.v_template = Variable(
            torch.Tensor(v_sym),
            requires_grad=False).to(device)

        # Regressor for joint locations given shape 
        self.J_regressor = Variable(
            torch.Tensor(dd['J_regressor'].T.todense()),
            requires_grad=False).to(device)

        # Pose blend shape basis
        num_pose_basis = dd['posedirs'].shape[-1]
        
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = Variable(
            torch.Tensor(posedirs), requires_grad=False).to(device)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = Variable(
            torch.Tensor(undo_chumpy(dd['weights'])),
            requires_grad=False).to(device)


    def __call__(self, beta, theta, trans=None, del_v=None, betas_logscale=None, get_skin=True, v_template=None):

        if True:
            nBetas = beta.shape[1]
        else:
            nBetas = 0

        
        # v_template = self.v_template.unsqueeze(0).expand(beta.shape[0], 3889, 3)
        if v_template is None:
            v_template = self.v_template

        # 1. Add shape blend shapes
        if nBetas > 0:
            if del_v is None:
                v_shaped = v_template + torch.reshape(torch.matmul(beta.to(self.device), self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
            else:
                v_shaped = v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = v_template.unsqueeze(0)
            else:
                v_shaped = v_template + del_v 

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)
        
        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        if len(theta.shape) == 4:
            Rs = theta
        else:
            Rs = torch.reshape( batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, 35, 3, 3])
        
        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(beta.device), [-1, 306])
        
        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        #4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(
            Rs, J, self.parents, betas_logscale=betas_logscale)


        # 5. Do skinning:
        num_batch = theta.shape[0]
        
        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])

            
        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])),
                [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat(
                [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=beta.device)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch,3)).to(device=beta.device)

        verts = verts + trans[:,None,:]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        joints = torch.cat([
            joints,
            verts[:, None, 1863], # end_of_nose 35
            verts[:, None, 26], # chin 36
            verts[:, None, 2124], # right ear tip 37
            verts[:, None, 150], # left ear tip 38
            verts[:, None, 3055], # left eye 39
            verts[:, None, 1097], # right eye 40
            verts[:, None, [1068, 1080, 1029, 1226]].mean(axis=2), #41
            verts[:, None, [2660, 3030, 2675, 3038]].mean(axis=2), #42
            verts[:, None, [910]].mean(axis=2), #43
            verts[:, None, [360, 1203, 1235, 1230]].mean(axis=2), #44
            verts[:, None, [3188, 3156, 2327, 3183]].mean(axis=2), #45
            verts[:, None, [1976, 1974, 1980, 856]].mean(axis=2), #46
            verts[:, None, [3854, 2820, 3852, 3858]].mean(axis=2), #47
            verts[:, None, [452, 1811]].mean(axis=2), #48
            verts[:, None, [416, 235, 182]].mean(axis=2), #49
            verts[:, None, [2156, 2382, 2203]].mean(axis=2), #50
            verts[:, None, [829]].mean(axis=2), #51
            verts[:, None, [2793]].mean(axis=2), #52
            verts[:, None, [60, 114, 186, 59]].mean(axis=2), #53
            verts[:, None, [2091, 2037, 2036, 2160]].mean(axis=2), #54
            verts[:, None, [384, 799, 1169, 431]].mean(axis=2), #55
            verts[:, None, [2351, 2763, 2397, 3127]].mean(axis=2), #56
            verts[:, None, [221, 104]].mean(axis=2), #57
            verts[:, None, [2754, 2192]].mean(axis=2), #58
            verts[:, None, [191, 1158, 3116, 2165]].mean(axis=2), #59
            verts[:, None, [28, 1109, 1110, 1111, 1835, 1836, 3067, 3068, 3069]].mean(axis=2), #60
            verts[:, None, [498, 499, 500, 501, 502, 503]].mean(axis=2), #61
            verts[:, None, [2463, 2464, 2465, 2466, 2467, 2468]].mean(axis=2), #62
            verts[:, None, [764, 915, 916, 917, 934, 935, 956]].mean(axis=2), #63
            verts[:, None, [2878, 2879, 2880, 2897, 2898, 2919, 3751]].mean(axis=2), #64
            verts[:, None, [1039, 1845, 1846, 1870, 1879, 1919, 2997, 3761, 3762]].mean(axis=2), #65
            verts[:, None, [0, 464, 465, 726, 1824, 2429, 2430, 2690]].mean(axis=2), #66
            ], dim = 1) 
        # print(joints.shape)
        if get_skin:
            return verts, joints, Rs, v_shaped
        else:
            return joints











