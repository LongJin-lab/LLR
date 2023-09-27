import argparse

class default_parser:
    def __init__(self) -> None:
        pass
    
    def wandb_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--wandb', action='store_true')
        parser.add_argument('--wandb_project', type=str, default='NeurIPs2022-Sparse SAM', help="Project name in wandb.")
        parser.add_argument('--wandb_name', type=str, default='Default', help="Experiment name in wandb.")
        return parser

    def base_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output_dir', type=str, default='/media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/', help='Name of dir where save all experiments.')#'/media_HDD_1/lab415/clm/deforming_the_loss_surface/FromCQserver/Sparse-Sharpness-Aware-Minimization-main/logs/'
        parser.add_argument('--output_name', type=str, default=None, help="Name of dir where save the log.txt&ckpt.pth of this experiment. (None means auto-set)")
        parser.add_argument('--resume', action='store_true', help="resume model,opt,etc.")
        parser.add_argument('--resume_path', type=str, default='.')

        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--log_freq', type=int, default=300, help="Frequency of recording information.")

        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--epochs', type=int, default=200, help="Epochs of training.")
        
        parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
        parser.add_argument('--cuda', '-c', action='store_true',  default=True, help='use cuda')
        parser.add_argument('--threads', default=4, type=int, help='number of threads')
        parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
        # data parameters
        parser.add_argument('--datapath', default='/media/bdc/clm/data/', metavar='DIR', help='path to the dataset')        
        parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
        parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
        parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
        parser.add_argument('--dir_type', default='states', help='direction type: weights | states (including BN\'s running_mean/var)')
        # parser.add_argument('--x', default='-0.2:0.2:51', help='A string with format xmin:x_max:xnum')#-0.2:0.2:17

        parser.add_argument('--x', default='-0.1:0.1:51', help='A string with format xmin:x_max:xnum')#-0.2:0.2:17
        parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
        parser.add_argument('--xnorm', default='filter', help='direction normalization: filter | layer | weight')
        parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
        parser.add_argument('--xignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
        parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
        parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
        parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
        parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
        parser.add_argument('--model_file', default='', help='path to the trained model file')
        parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
        parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
        parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')
        parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
        parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')
        parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
        # parser.add_argument('--loss_max', default=1.0, type=float, help='Maximum value to show in 1D plot')
        parser.add_argument('--loss_max', default=0.002, type=float, help='Maximum value to show in 1D plot')#0.02#3

        parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
        parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
        parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
        parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
        parser.add_argument('--logscale', action='store_true', default=False, help='use log scale for loss values')
        parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

        
        
        
        return parser

    def dist_parser(self):
        parser = argparse.ArgumentParser(add_help=False)    
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        return parser


    def data_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--dataset', type=str, default='CIFAR10_base', help="Dataset name in `DATASETS` registry.")
        parser.add_argument('--datadir', type=str, default='/media/bdc/clm/data/', help="Path to your dataset.")#/media_SSD_1/datasets/
        parser.add_argument('--batch_size', type=int, default=128, help="Batch size used in training and validation.")
        parser.add_argument('--num_workers', type=int, default=4, help="Number of CPU threads for dataloaders.")
        parser.add_argument('--pin_memory', action='store_true', default=True)
        parser.add_argument('--drop_last', action='store_true', default=True)
        parser.add_argument('--distributed_val', action='store_true', help="Enabling distributed evaluation (Only works when use multi gpus).")
        parser.add_argument('--val_rat', type=float, default=0.0, help="")

        return parser

    def base_opt_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--opt', type=str, default='sgd')
        parser.add_argument('--lr', type=float, default=0.05)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        # sgd
        parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD.(None means the default in optm)")
        parser.add_argument('--nesterov', action="store_true")
        # adam
        parser.add_argument('--betas', type=float, default=None, nargs='+', help="Betas for AdamW Optimizer.(None means the default in optm)")
        parser.add_argument('--eps', type=float, default=None, help="Epsilon for AdamW Optimizer.(None means the default in optm)")
        parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
        parser.add_argument('--swa_lr', type=float, default=None, metavar='SWALR', help='SWA LR (default: 0.05)')
        # parser.add_argument('--dls_coe', default=[1., 5.], nargs='+', type=float)
        parser.add_argument('--dls_coe0', default=None, type=float)
        parser.add_argument('--dls_coe1', default=None, type=float)
        parser.add_argument('--dls_coe2', default=None, type=float)
        
        parser.add_argument('--dls_list', default=None, type=str)

        parser.add_argument('--dls_act', type=str, default=None, metavar='DLSACT', help='SWA LR (default: 0.05)')
        parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
        parser.add_argument('--pid', action='store_true', help='swa usage flag (default: off)')
        parser.add_argument('--tuning', action='store_true', help='swa usage flag (default: off)')
        parser.add_argument('--norm_loss', action='store_true', help='swa usage flag (default: off)')
        # parser.add_argument('--n_inter', type=int, default=None, metavar='N', help='evaluation frequency (default: 5)')
        # parser.add_argument('--inter_start', type=int, default=None, metavar='N', help='evaluation frequency (default: 5)')
        parser.add_argument('--dls_start', type=float, default=None,  help='SWA LR (default: 0.05)')
        parser.add_argument('--dls_end', type=float, default=None, help='SWA LR (default: 0.05)')
        parser.add_argument('--val_type', type=str, default='train', help='SWA LR (default: 0.05)')
        parser.add_argument('--setting', type=str, default='Default', help='SWA LR (default: 0.05)')
        parser.add_argument('--rep_num', type=int, default=3, metavar='N', help='evaluation frequency (default: 5)')
        parser.add_argument('--std', type=float, default=0.002, help='standard deviation')
        parser.add_argument('--M', type=int, default=1, help='M')
        parser.add_argument('--smooth_out_a', type=float, default=0.009, help='standard deviation')
        parser.add_argument('--plot_grad', action='store_true', help='standard deviation')
        parser.add_argument('--plot_surf', action='store_true', help='standard deviation')
        parser.add_argument('--end', default=1.0, type=float, help='Miminum value to map')
        parser.add_argument('--plus_base', default=None, type=float, help='Miminum value to map')
        parser.add_argument('--min_shift', default=None, type=bool, help='Miminum value to map')
        parser.add_argument('--calm_base', default=1.0, type=float, help='Miminum value to map')
        parser.add_argument('--calm_epoch', default=10, type=int, help='Miminum value to map')
        parser.add_argument('--StartLast', default=None, type=int, help='Miminum value to map')

        return parser

    def sam_opt_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--rho', type=float, default=0.05, help="Perturbation intensity of SAM type optims.")
        parser.add_argument('--sparsity', type=float, default=0.2, help="The proportion of parameters that do not calculate perturbation.")
        parser.add_argument('--update_freq', type=int, default=5, help="Update frequency (epoch) of sparse SAM.")

        parser.add_argument('--num_samples', type=int, default=1024, help="Number of samples to compute fisher information. Only for `ssam-f`.")
        parser.add_argument('--drop_rate', type=float, default=0.5, help="Death Rate in `ssam-d`. Only for `ssam-d`.")
        parser.add_argument('--drop_strategy', type=str, default='gradient', help="Strategy of Death. Only for `ssam-d`.")
        parser.add_argument('--growth_strategy', type=str, default='random', help="Only for `ssam-d`.")
        return parser

    def lr_scheduler_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--warmup_epoch', type=int, default=0)
        parser.add_argument('--warmup_init_lr', type=float, default=0.0)
        parser.add_argument('--lr_scheduler', type=str, default='CosineLRscheduler')
        # CosineLRscheduler
        parser.add_argument('--eta_min', type=float, default=0)
        # MultiStepLRscheduler
        parser.add_argument('--milestone', type=int, nargs='+', default=[100,150,200], help="Milestone for MultiStepLRscheduler.")#[60, 120, 160]
        parser.add_argument('--gamma', type=float, default=0.1, help="Gamma for MultiStepLRscheduler.")
        return parser

    def model_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--model', type=str, default='resnet18', help="Model in registry to use.")
        return parser


    def get_args(self):
        all_parser_funcs = []
        for func_or_attr in dir(self):
            if callable(getattr(self, func_or_attr)) and not func_or_attr.startswith('_') and func_or_attr[-len('parser'):] == 'parser':
                # print('func_or_attr',func_or_attr)
                all_parser_funcs.append(getattr(self, func_or_attr))
        all_parsers = [parser_func() for parser_func in all_parser_funcs]
        # print('all_parsers',all_parsers)

        final_parser = argparse.ArgumentParser(parents=all_parsers)
        # print('final_parser',final_parser)

        args = final_parser.parse_args()
        # print('args',args)

        self.auto_set_name(args)
        return args

    def auto_set_name(self, args):



        if args.wandb_name == 'Default':
            args.wandb_name = args.output_name