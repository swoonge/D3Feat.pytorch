import argparse
import time
import os

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')

# Network configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--num_layers', type=int, default=5)
net_arg.add_argument('--in_points_dim', type=int, default=3)
net_arg.add_argument('--first_features_dim', type=int, default=128)
net_arg.add_argument('--first_subsampling_dl', type=float, default=0.3) # First size of grid used for subsampling
net_arg.add_argument('--in_features_dim', type=int, default=1) 
net_arg.add_argument('--in_radius', type=float, default=15.0)
net_arg.add_argument('--conv_radius', type=float, default=2.5)
net_arg.add_argument('--deform_radius', type=float, default=6.0)
net_arg.add_argument('--density_parameter', type=float, default=3.0) # previous name for deform_radius
net_arg.add_argument('--num_kernel_points', type=int, default=15)
net_arg.add_argument('--KP_extent', type=float, default=1.2)
net_arg.add_argument('--KP_influence', type=str, default='linear')
net_arg.add_argument('--convolution_mode', type=str, default='sum', choices=['closest', 'sum'])
net_arg.add_argument('--aggregation_mode', type=str, default='sum', choices=['closest', 'sum'])
net_arg.add_argument('--fixed_kernel_points', type=str, default='center', choices=['center', 'verticals', 'none'])
net_arg.add_argument('--use_batch_norm', type=str2bool, default=True)
net_arg.add_argument('--batch_norm_momentum', type=float, default=0.98)
net_arg.add_argument('--deformable', type=str2bool, default=False)
net_arg.add_argument('--modulated', type=str2bool, default=False)
 
# Loss configurations
loss_arg = add_argument_group('Loss')
loss_arg.add_argument('--dist_type', type=str, default='euclidean')
loss_arg.add_argument('--desc_loss', type=str, default='circle', choices=['contrastive', 'circle'])
loss_arg.add_argument('--pos_margin', type=float, default=0.1)
loss_arg.add_argument('--neg_margin', type=float, default=1.4)
loss_arg.add_argument('--m', type=float, default=0.1)
loss_arg.add_argument('--log_scale', type=float, default=10)
loss_arg.add_argument('--safe_radius', type=float, default=3.0)
loss_arg.add_argument('--det_loss', type=str, default='score')
loss_arg.add_argument('--desc_loss_weight', type=float, default=1.0)
loss_arg.add_argument('--det_loss_weight', type=float, default=1.0)

# Optimizer configurations
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'ADAM'])
opt_arg.add_argument('--max_epoch', type=int, default=300)
opt_arg.add_argument('--training_max_iter', type=int, default=3500)
opt_arg.add_argument('--val_max_iter', type=int, default=200)
opt_arg.add_argument('--lr', type=float, default=0.01)
opt_arg.add_argument('--weight_decay', type=float, default=0.00001)
opt_arg.add_argument('--momentum', type=float, default=0.98)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--scheduler_gamma', type=float, default=0.1**(1/80))
opt_arg.add_argument('--scheduler_interval', type=int, default=1)
opt_arg.add_argument('--grad_clip_norm', type=float, default=100.0)

# Dataset and dataloader configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--root', type=str, default='./data/kitti_map/')
data_arg.add_argument('--num_node', type=int, default=128)
data_arg.add_argument('--downsample', type=float, default=0.3)
data_arg.add_argument('--self_augment', type=str2bool, default=True)
data_arg.add_argument('--augment_noise', type=float, default=0.001)
data_arg.add_argument('--augment_axis', type=int, default=1)
data_arg.add_argument('--augment_scale_min', type=float, default=0.9)
data_arg.add_argument('--augment_scale_max', type=float, default=1.1)
data_arg.add_argument('--augment_rotation', type=float, default=1.0, help='rotation angle = num * 2pi') 
data_arg.add_argument('--augment_translation', type=float, default=0.05, help='translation = num (m)')
data_arg.add_argument('--batch_size', type=int, default=1)
data_arg.add_argument('--num_workers', type=int, default=0)

# Other configurations
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--gpu_mode', type=str2bool, default=True)
misc_arg.add_argument('--verbose', type=str2bool, default=True)
misc_arg.add_argument('--pretrain', type=str, default='')

experiment_id = "kitti_map" + time.strftime('%m%d%H%M')
# snapshot configurations
snapshot_arg = add_argument_group('Snapshot')
snapshot_arg.add_argument('--snapshot_dir', type=str, default=f'./data/kitti_map/snapshot/{experiment_id}')
snapshot_arg.add_argument('--tboard_dir', type=str, default=f'./data/kitti_map/tensorboard/{experiment_id}')
snapshot_arg.add_argument('--snapshot_interval', type=int, default=100)
snapshot_arg.add_argument('--save_dir', type=str, default=os.path.join(f'./data/kitti_map/snapshot/{experiment_id}', 'models/'))

def get_config():
  args = parser.parse_args()
  return args
