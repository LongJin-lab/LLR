"""
    1D plotting routines
"""

from matplotlib import pyplot as pp
import h5py
import argparse
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick

matplotlib.use('Agg')

pp.style.use(['ieee'])
# plt.style.use(['seaborn-paper'])
matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
        }
)
    
def plot_1d_loss_err(surf_file, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=False):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    print(f.keys())
    x = f['xcoordinates'][:]
    assert 'train_loss' in f.keys(), "'train_loss' does not exist"
    train_loss = f['train_loss'][:]
    train_acc = f['train_acc'][:]

    print("train_loss")
    print(train_loss)
    print("train_acc")
    print(train_acc)

    # xmin = xmin if xmin <= -1.0 else min(x)
    # xmax = xmax if xmax <= 1.0 else max(x)

    # loss and accuracy map
    fig, ax1 = pp.subplots(sharex=True)
    ax2 = ax1.twinx()
    if log:
        tr_loss, = ax1.semilogy(x, train_loss, 'b-', label='Training loss', linewidth=1)
    else:
        tr_loss, = ax1.plot(x, train_loss, color='indianred', linestyle = '-', label='Training loss', linewidth=2)
    tr_acc, = ax2.plot(x, train_acc, color='midnightblue', linestyle = '--', label='Training accuracy', linewidth=1)

    if 'test_loss' in f.keys():
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]
        print("test_loss")
        print(test_loss)
        print("test_acc")
        print(test_acc)        
        if log:
            te_loss, = ax1.semilogy(x, test_loss, 'b--', label='Test loss', linewidth=1)
        else:
            te_loss, = ax1.plot(x, test_loss, 'b--', label='Test loss', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Test accuracy', linewidth=1)

    pp.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='indianred', fontsize='xx-large')
    ax1.tick_params('y', colors='indianred', labelsize='x-large')
    ax1.tick_params('x', labelsize='xx-large')
    # ax1.set_ylim(0, loss_max)
    ax1.set_ylim(min(train_loss)-0.0002,min(train_loss)+loss_max)#0.0035, 0.0045
    ax2.set_ylabel('Accuracy', color='midnightblue', fontsize='xx-large')
    ax2.tick_params('y', colors='midnightblue', labelsize='x-large')
    # ax2.set_ylim(0, 100)
    ax2.set_ylim(max(train_acc)-0.02,max(train_acc))#0.0035, 0.0045
    ax1.set_xlabel(r'$\alpha$', fontsize='xx-large')
    pp.tight_layout()
    pp.savefig(surf_file + '_1d_Trainingloss_acc' + ('_log' if log else '') + '.pdf',
                dpi=300, bbox_inches='tight', format='pdf')

    pp.close(fig)
    
    # train_loss curve
    pp.figure()
    
    fig, (ax1, ax2) = pp.subplots(2, 1, sharex=True)
    # fig.subplots_adjust()
    fig.subplots_adjust(hspace=0.05)
    # fig.subplots_adjust(hspace=0.05)  # adjust space between axes
    if 'test_loss' in f.keys():
        if log:
            ax1.semilogy(x, test_loss, '--', label='Test', color='steelblue')
        else:
            ax1.plot(x, test_loss, '--', label='Test', color='steelblue')    
    if log:
        ax2.semilogy(x, train_loss, label='Training',color='indianred')
    else:
        ax2.plot(x, train_loss, label='Training',color='indianred')


    # set the y-limits for each axis
    # ax1.set_ylim(min(test_loss)*0.95, max(test_loss)*1.05)
    # ax2.set_ylim(min(train_loss)*0.95, max(train_loss)*1.05)
    # ax1.set_ylim(min(test_loss)-0.002, min(test_loss)+0.02)#0.91, 0.92#1.08, 1.09

    # ax2.set_ylim(min(train_loss),min(train_loss)+0.002)#0.0035, 0.0045
    # pp.xlim(xmin, xmax)
    # ax2.set_ylim(min(train_loss),min(train_loss)+0.02)#0.0035, 0.0045
    ax1.set_ylim(min(test_loss)-0.002, min(test_loss)+5*loss_max)#0.91, 0.92#1.08, 1.09
    # ax1.set_ylim(min(test_loss)-0.01*loss_max, min(test_loss)+loss_max)#0.91, 0.92#1.08, 1.09
    ax2.set_ylim(min(train_loss)-0.0002,min(train_loss)+loss_max)#0.0035, 0.0045

    pp.xlim(xmin, xmax)
    # hide the spines between the axes
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # ax1.spines['left'].set_bounds(0, 2)
    # ax1.spines['right'].set_bounds(8, 10)
    # ax1.spines['left'].set_linestyle('--')
    # ax1.spines['right'].set_linestyle('--')
    # ax2.spines['left'].set_linestyle('--')
    # ax2.spines['right'].set_linestyle('--')
    # create the slanted lines
    d = .015  # controls the length and angle of the lines
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    # pp.legend(fontsize='xx-large')
    ax1.legend(loc='upper right',fontsize='xx-large')
    ax2.legend(loc='upper right',fontsize='xx-large')


    # fig.text(0, 0.5, 'Loss', va='center', rotation='vertical',fontsize='xx-large')

    # pp.ylabel('Loss', fontsize='xx-large')
    pp.xlabel(r'$\alpha$', fontsize='xx-large')
    ax1.tick_params( 'both', labelsize='x-large')
    ax2.tick_params( 'both', labelsize='x-large')  
    # ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    # ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    # ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    # ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))      
    fig.supylabel('Loss', va='center', rotation='vertical', fontsize='xx-large')
    # fig.subplots_adjust(left=0.2)
    pp.tight_layout()

    # pp.ylim(0, loss_max)
    pp.savefig(surf_file + '_1d_loss' + ('_log' if log else '') +'_'+str(xmin)+'_'+str(xmax)+'.pdf',
                dpi=300, bbox_inches='tight', format='pdf')
    pp.close()
    
    
    # train_err curve
    pp.figure()
    pp.plot(x, 100 - train_acc)
    pp.xlim(xmin, xmax)
    pp.ylim(0, 100)
    pp.ylabel('Training Error', fontsize='xx-large')
    pp.savefig(surf_file + '_1d_train_err.pdf', dpi=300, bbox_inches='tight', format='pdf')
    pp.close()

    # train_loss curve
    pp.figure()
    pp.plot(x, train_loss, color='indianred')
    pp.xlim(xmin, xmax)
    pp.ylim(min(train_loss)-0.1,max(train_loss)+0.1)
    pp.ylabel('Training loss', fontsize='xx-large')
    pp.xlabel(r'$\alpha$', fontsize='xx-large')
    pp.tick_params( 'both', labelsize='x-large')
    pp.savefig(surf_file + '_1d_train_loss.pdf', dpi=300, bbox_inches='tight', format='pdf')
    pp.close()
    pp.close('all')
    #if show: pp.show()
    f.close()

def plot_1d_loss_appro(surf_file, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=False):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    print(f.keys())
    x = f['xcoordinates'][:]
    assert 'train_loss' in f.keys(), "'train_loss' does not exist"
    train_loss = f['train_loss'][:]
    train_acc = f['train_acc'][:]

    print("train_loss")
    print(train_loss)
    print("train_acc")
    print(train_acc)

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 1.0 else max(x)
    # xmin = -1
    # xmax = 1
    x_mask = np.logical_and(x >= xmin, x <= xmax)
    x = x[x_mask]
    train_loss = train_loss[x_mask]
    def plot_appro(type, rat, color):
        if type == 'High':
            start_idx  = int(len(x) * rat)
            end_idx = int(len(x) * (1-rat))
            x_range = [0, start_idx, end_idx, -1]
        elif type == 'Low':
            start_idx  = int(len(x) * rat)
            end_idx = int(len(x) * (1-rat))
            x_range = [start_idx, len(x) // 2, len(x) // 2+1, end_idx]
        mask = np.zeros_like(x, dtype=bool)
        for i in range(len(x_range) // 2):
            start_idx = x_range[i * 2]
            end_idx = x_range[i * 2 + 1]
            mask[start_idx:end_idx] = True
        y_range = [np.min(train_loss[mask]), np.max(train_loss[mask])]
        # Generate the quadratic approximation
        p = np.polyfit(x[mask], train_loss[mask], 2)
        y_approx = np.polyval(p, x)
        # pp.plot(x, y_approx, label=type+'-loss quadratic',alpha=0.5,color=color, linewidth=3)
        # Highlight the x-range being considered
        for i in range(len(x_range) // 2):
            start = x[x_range[i * 2]]
            end = x[x_range[i * 2 + 1]]
            # pp.axvspan(start, end, alpha=0.2, color='gray')
        pp.fill_between(x, y_range[0], y_range[1], alpha=0.1, color=color)         
        return y_approx
    # train_loss curve
    pp.figure()
    pp.plot(x, train_loss, color='indianred', linewidth=2,label='Exact loss')
    
    # y_approx1 = plot_appro('High', 0.1, 'slateblue')
    # y_approx2 = plot_appro('Low', 0.2, 'teal')


    pp.xlim(xmin, xmax)
    # y_min = np.min(np.minimum(np.minimum(train_loss, y_approx1), y_approx2))
    # y_max = np.max(np.maximum(np.maximum(train_loss, y_approx1), y_approx2))
    pp.ylim(min(train_loss),min(train_loss)+loss_max/2)#0.0035, 0.0045

    # print('(y_min, y_max)',(y_min, y_max))
    # pp.ylim(y_min, y_max)
    pp.ylabel('Training loss', fontsize='xx-large')
    pp.xlabel(r'$\alpha$', fontsize='xx-large')
    pp.tick_params( 'both', labelsize='x-large')
    # pp.legend(fontsize='x-large')
    pp.tight_layout()
    pp.savefig(surf_file + '_1d_train_loss.pdf', dpi=300, bbox_inches='tight', format='pdf')
    pp.close()
    pp.close('all')
    #if show: pp.show()
    f.close()
    
def plot_1d_loss_err_repeat(prefix, idx_min=1, idx_max=10, xmin=-1.0, xmax=1.0,
                            loss_max=5, show=False):
    """
        Plotting multiple 1D loss surface with different directions in one figure.
    """

    fig, ax1 = pp.subplots()
    ax2 = ax1.twinx()

    for idx in range(idx_min, idx_max + 1):
        # The file format should be prefix_{idx}.h5
        f = h5py.File(prefix + '_' + str(idx) + '.h5','r')

        x = f['xcoordinates'][:]
        train_loss = f['train_loss'][:]
        train_acc = f['train_acc'][:]
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]

        xmin = xmin if xmin != -1.0 else min(x)
        xmax = xmax if xmax != 1.0 else max(x)

        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
        te_loss, = ax1.plot(x, test_loss, 'b--', label='Testing loss', linewidth=1)
        tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Testing accuracy', linewidth=1)

    pp.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    ax1.set_ylim(0, loss_max)
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    ax2.set_ylim(0, 100)
    pp.savefig(prefix + '_1d_loss_err_repeat.pdf', dpi=300, bbox_inches='tight', format='pdf')
    pp.close()
    #if show: pp.show()


def plot_1d_eig_ratio(surf_file, xmin=-1.0, xmax=1.0, val_1='min_eig', val_2='max_eig', ymax=1, show=False):
    print('------------------------------------------------------------------')
    print('plot_1d_eig_ratio')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    x = f['xcoordinates'][:]

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])
    abs_ratio = np.absolute(np.divide(Z1, Z2))

    pp.plot(x, abs_ratio)
    pp.xlim(xmin, xmax)
    pp.ylim(0, ymax)
    pp.savefig(surf_file + '_1d_eig_abs_ratio.pdf', dpi=300, bbox_inches='tight', format='pdf')

    ratio = np.divide(Z1, Z2)
    pp.plot(x, ratio)
    pp.xlim(xmin, xmax)
    pp.ylim(0, ymax)
    pp.savefig(surf_file + '_1d_eig_ratio.pdf', dpi=300, bbox_inches='tight', format='pdf')
    pp.close()

    f.close()
    #if show: pp.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plott 1D loss and error curves')
    parser.add_argument('--surf_file', '-f', default='/media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_base_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.0-0.1_-1.03None1.0None2023-08-19-14:43:36/199_-0.1_-1.0_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.1,0.1,51].h5')
    parser.add_argument('--log', action='store_true', default=None, help='logarithm plot')
    parser.add_argument('--xmin', default=-0.1, type=float, help='xmin value')
    parser.add_argument('--xmax', default=0.1, type=float, help='xmax value')
    parser.add_argument('--loss_max', default=0.002, type=float, help='ymax value')#0.02
    parser.add_argument('--show', action='store_true', default=False, help='show plots')
    parser.add_argument('--prefix', default='', help='The common prefix for surface files')
    parser.add_argument('--idx_min', default=1, type=int, help='min index for the surface file')
    parser.add_argument('--idx_max', default=10, type=int, help='max index for the surface file')

    args = parser.parse_args()

    if args.prefix:
        plot_1d_loss_err_repeat(args.prefix, args.idx_min, args.idx_max,
                                args.xmin, args.xmax, args.loss_max, args.show)
    else:
        plot_1d_loss_err(args.surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
        plot_1d_loss_appro(args.surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
        # plot_1d_eig_ratio(args.surf_file, args.xmin, args.xmax, val_1='min_eig', val_2='max_eig', ymax=1, show=False)