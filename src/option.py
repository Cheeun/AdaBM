import argparse

parser = argparse.ArgumentParser(description='EDSR and MDSR')


parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../dataset',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
                    
# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
# this is same as iters per batch
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--step', type=int, default='1',
                    help='learning rate step size')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='learning rate decay factor for step decay')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='', 
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

parser.add_argument('--quantize_a', type=float, default=32, help='activation_bit')
parser.add_argument('--quantize_w', type=float, default=32, help='weight_bit')

parser.add_argument('--batch_size_calib', type=int, default=16, help='input batch size for calib')
parser.add_argument('--batch_size_update', type=int, default=2, help='input batch size for update')
parser.add_argument('--num_data', type=int, default=800, help='number of data for PTQ')

parser.add_argument('--fq', action='store_true', help='fully quantize')
parser.add_argument('--imgwise', action='store_true', help='imgwise')
parser.add_argument('--layerwise', action='store_true', help='layerwise')
parser.add_argument('--bac', action='store_true', help='bac')

parser.add_argument('--lr_w', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_a', type=float, default=0.05, help='learning rate')
parser.add_argument('--lr_measure_layer', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_measure_img', type=float, default=0.01, help='learning rate')
parser.add_argument('--w_bitloss', type=float, default=50.0, help='weight for bit loss')
parser.add_argument('--w_sktloss', type=float, default=10.0, help='weight for skt loss')

parser.add_argument('--test_patch', action='store_true', help='test patch')
parser.add_argument('--test_patch_size', type=int, default=96, help='test patch size')
parser.add_argument('--test_step_size', type=int, default=90, help='test step size')
parser.add_argument('--test_own', type=str, default=None, help='directory for own test image')
parser.add_argument('--n_parallel', type=int, default=1, help='number of patches for parallel processing')

parser.add_argument('--quantizer', default='minmax', choices=('minmax', 'percentile', 'omse'), help='quantizer to use')
parser.add_argument('--quantizer_w', default='minmax', choices=('minmax', 'percentile', 'omse'), help='quantizer to use')
parser.add_argument('--percentile_alpha', type=float, default=0.99, help='used when quantizer is percentile')

parser.add_argument('--ema_beta', type=float, default=0.9, help='beta for EMA')
parser.add_argument('--bac_beta', type=float, default=0.5, help='beta for EMA in BaC')

parser.add_argument('--img_percentile', type=float, default=10.0, help='clip percentile for u,l that ranges from 0~100')
parser.add_argument('--layer_percentile', type=float, default=30.0, help='clip percentile for u,l that ranges from 0~100')

args = parser.parse_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

