"""Two preregistered full64 assignment transfer layouts; original objective retained."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def matrices(predicted,goal):
    costs=(predicted[..., :,None,:]-goal[...,None,:,:]).square().sum(-1)
    return costs,costs.reshape(-1,*costs.shape[-2:])


def bulk_original_sum(predicted,goal):
    costs,flat=matrices(predicted,goal);host=flat.detach().cpu().numpy();values=[]
    for matrix,cpu in zip(flat,host):
        row,col=linear_sum_assignment(cpu);values.append(matrix[row,col].sum())
    return torch.stack(values).reshape(costs.shape[:-2])


def bulk_batched_sum(predicted,goal):
    costs,flat=matrices(predicted,goal);host=flat.detach().cpu().numpy()
    assignments=[linear_sum_assignment(matrix) for matrix in host]
    rows=torch.as_tensor(np.stack([a[0] for a in assignments]),device=flat.device)
    cols=torch.as_tensor(np.stack([a[1] for a in assignments]),device=flat.device)
    batch=torch.arange(len(flat),device=flat.device)[:,None]
    return flat[batch,rows,cols].sum(-1).reshape(costs.shape[:-2])

VARIANTS={'bulk_cost_copy_original_per_matrix_GPU_sum':bulk_original_sum,
          'bulk_cost_copy_batched_indices_GPU_sum':bulk_batched_sum}
