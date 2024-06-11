import torch
import torch.nn as nn
import numpy as np
from .evals.al_model_evaluation import compute_ray_errors

def get_grid_mask(points, pc_range):
    points = points.T
    mask1 = np.logical_and(pc_range[0] <= points[0], points[0] <= pc_range[3])
    mask2 = np.logical_and(pc_range[1] <= points[1], points[1] <= pc_range[4])
    mask3 = np.logical_and(pc_range[2] <= points[2], points[2] <= pc_range[5])

    mask = mask1 & mask2 & mask3

    return mask

def get_clamped_gt(n_output, points, tindex, pc_range,
        eval_within_grid=False, eval_outside_grid=False, get_indices=False):
    pcds = []
    if get_indices:
        indices = []
    for t in range(n_output):
        mask = tindex == t
        if eval_within_grid:
            mask = np.logical_and(mask, get_grid_mask(points, pc_range))
        if eval_outside_grid:
            mask = np.logical_and(mask, ~get_grid_mask(points, pc_range))
        # skip the ones with no data
        if not mask.any():
            continue
        if get_indices:
            idx = np.arange(points.shape[0])
            indices.append(idx[mask])
        gt_pts = points[mask, :3]
        pcds.append(torch.from_numpy(gt_pts))
    if get_indices:
        return pcds, indices
    else:
        return pcds

class RayTracing_Model(nn.Module):
    def __init__(self, configs, **kwargs) -> None:
        super(RayTracing_Model, self).__init__()
        self.input_T = configs.data_config['n_input']
        self.output_T = configs.data_config['n_output']
        self.pc_range = configs.data_config['pc_range']

    def forward(self, input_points,
                      input_tindex,
                      output_origin,
                      output_points,
                      output_tindex,
                      output_labels,
                      eval_within_grid=False,
                      eval_outside_grid=False,
                      **kwargs):
        
        inference_dict = {}
        inference_dict['pred_pcd'] = []

        for j in range(output_points.shape[0]):  # iterate through the batch
            batch_pred_pcd = []
            pred_pcd_agg = input_points[j].cpu()

            gt_pcds = get_clamped_gt(
                    self.output_T,
                    output_points[j].cpu().numpy(),
                    output_tindex[j].cpu().numpy(),
                    self.pc_range ,
                    eval_within_grid,
                    eval_outside_grid
                )
            
            # load predictions
            for k in range(len(gt_pcds)):
                origin = output_origin[j][k].cpu().numpy()
                gt_pcd = gt_pcds[k]
                pred_pcd = compute_ray_errors(
                        pred_pcd_agg,
                        gt_pcd,
                        torch.from_numpy(origin),
                        output_origin.device,
                        return_interpolated_pcd=True)
                batch_pred_pcd.append(pred_pcd)
            inference_dict['pred_pcd'].append(batch_pred_pcd)

        if 'inference_mode' in kwargs:
            return inference_dict