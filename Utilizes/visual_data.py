"""
-- coding: utf-8 --
@Time : 2022/07/24 14:20
@Author : Tianyuan Liu
@Department : Baidu ACG
@File : visual_data.py
"""
import sys
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sbn
from scipy import stats
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri
import matplotlib.cm as cm
import os
from matplotlib import ticker, rcParams
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl


def adjacent_values(vals, q1, q3):
    """
    生成四分点，plot_violin
    """
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels, position=None):
    """
    生成四分点，plot_violin
    """
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    if position is None:
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xlim(0.25, len(labels) + 0.75)
    else:
        ax.set_xticks(position)
        ax.set_xlim(1.5 * position[0] - 0.5 * position[1], 1.5 * position[-1] - 0.5 * position[-2])
    ax.set_xticklabels(labels)


class TextLogger(object):
    """
    log文件记录所有打印结果
    """

    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        """
        文本输出记录
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        通过
        """
        pass


class MatplotlibVision(object):

    def __init__(self, log_dir, input_name=('x'), field_name=('f',)):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        # sbn.set_style('ticks')
        # sbn.set()

        self.field_name = field_name
        self.input_name = input_name

        self.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        self.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 20}
        self.box_line_width = 1.5
        self.font_size_label = 108
        self.font_size_cb = 72
        # self._cbs = [None] * len(self.field_name) * 3
        # gs = gridspec.GridSpec(1, 1)
        # gs.update(top=0.95, bottom=0.07, left=0.1, right=0.9, wspace=0.5, hspace=0.7)
        # gs_dict = {key: value for key, value in gs.__dict__.items() if key in gs._AllowedKeys}
        # self.fig, self.axes = plt.subplots(len(self.field_name), 3, gridspec_kw=gs_dict, num=100, figsize=(30, 20))
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        self.config = {"font.family": 'Times New Roman',
                       "font.size": 20,
                       "mathtext.fontset": 'stix',
                       "font.serif": ['SimSun'], }
        rcParams.update(self.config)

    def plot_loss(self, fig, axs, x, y, label, title=None, xylabels=['epoch', 'loss value']):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        axs.semilogy(x, y, label=label)
        axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)
        # plt.pause(0.001)

    def plot_value(self, fig, axs, x, y, label, title=None, xylabels=['x', 'y']):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        axs.plot(x, y, label=label)
        axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)

    def plot_scatter(self, fig, axs, true, pred, axis=0, title=None, xylabels=['x', 'y']):
        # sbn.set(color_codes=True)

        axs.scatter(np.arange(true.shape[0]), true, marker='*')
        axs.scatter(np.arange(true.shape[0]), pred, marker='.')

        axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)

    def plot_regression(self, fig, axs, true, pred, title=None, xylabels=['true value', 'pred value']):
        # 所有功率预测误差与真实结果的回归直线
        # sbn.set(color_codes=True)

        max_value = max(true)  # math.ceil(max(true)/100)*100
        min_value = min(true)  # math.floor(min(true)/100)*100
        split_value = np.linspace(min_value, max_value, 11)

        split_dict = {}
        split_label = np.zeros(len(true), np.int)
        for i in range(len(split_value)):
            split_dict[i] = str(split_value[i])
            index = true >= split_value[i]
            split_label[index] = i + 1

        axs.scatter(true, pred, marker='.')

        axs.plot([min_value, max_value], [min_value, max_value], 'r-', linewidth=5.0)
        axs.fill_between([min_value, max_value], [0.95 * min_value, 0.95 * max_value],
                         [1.05 * min_value, 1.05 * max_value],
                         alpha=0.2, color='b')

        # plt.ylim((min_value, max_value))
        axs.set_xlim((min_value, max_value))
        axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)

        # plt.ylim((-0.2, 0.2))
        # plt.pause(0.001)

    def plot_error(self, fig, axs, error, title=None,
                   xylabels=['predicted relative error / %', 'distribution density']):
        # sbn.set_color_codes()
        # ax.bar(np.arange(len(error)), error*100, )

        error = pd.DataFrame(error) * 100
        sbn.distplot(error, bins=20, norm_hist=True, rug=True, fit=stats.norm, kde=False,
                     rug_kws={"color": "g"}, fit_kws={"color": "r", "lw": 3}, hist_kws={"color": "b"})
        # plt.xlim([-1, 1])
        axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)

    def plot_fields1d(self, fig, axs, real, pred, coord, title=None, xylabels=['x coordinate', 'field'],
                      show_channel=None):

        if len(axs.shape) == 1:
            axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(len(self.field_name))

        num_channel = len(show_channel)
        name_channel = [self.field_name[i] for i in show_channel]

        for i in range(num_channel):

            fi = show_channel[i]
            ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
            limit = max(abs(ff[-1].min()), abs(ff[-1].max()))

            axs[i][0].cla()
            axs[i][0].plot(coord, ff[0], color='g', linewidth=3, label='true')
            axs[i][0].plot(coord, ff[1], '--', color='r', linewidth=2, label='pred')
            axs[i][1].plot(coord, ff[2], color='r', linewidth=2, label='error')
            for j in range(2):
                axs[i][j].legend(loc="best", prop=self.font)
                axs[i][j].set_xlabel(xylabels[0], fontdict=self.font)
                axs[i][j].set_ylabel(xylabels[1], fontdict=self.font)
                axs[i][j].tick_params('both', labelsize=self.font["size"], )
                axs[i][j].set_title(title, fontdict=self.font)

    def plot_fields_tr(self, fig, axs, real, pred, coord, edges, mask=None, cmin_max=None, fmin_max=None,
                       show_channel=None):

        if len(axs.shape) == 1:
            axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(len(self.field_name))

        name_channel = [self.field_name[i] for i in show_channel]

        if fmin_max is None:
            fmin, fmax = real.min(axis=0), real.max(axis=0)
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max is None:
            cmin, cmax = coord.min(axis=0), coord.max(axis=0)
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        titles = ['truth', 'predicted', 'error']
        cmaps = ['RdYlBu_r', 'RdYlBu_r', 'coolwarm']
        num_channel = len(show_channel)

        x = coord[:, 0]
        y = coord[:, 1]

        for i in range(num_channel):
            fi = show_channel[i]
            ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
            limit = max(abs(ff[-1].min()), abs(ff[-1].max()))
            for j in range(3):
                f_true = axs[i][j].tripcolor(x, y, ff[j], triangles=edges, cmap=cmaps[j], shading='gouraud',
                                             antialiased=True, snap=True)

                # f_true = axs[i][j].tricontourf(triObj, ff[j], 20, cmap=cmaps[j])
                if mask is not None:
                    axs[i][j].fill(mask[:, 0], mask[:, 1], facecolor='white')
                # f_true.set_zorder(10)

                # axs[i][j].grid(zorder=0, which='both', color='grey', linewidth=1)
                axs[i][j].set_title(titles[j], fontdict=self.font_CHN)
                axs[i][j].axis([cmin[0], cmax[0], cmin[1], cmax[1]])
                # axs[i][j].tick_params(axis='x', labelsize=)
                # if i == 0:
                #     ax[i][j].set_title(titles[j], fontdict=self.font_CHN)
                cb = fig.colorbar(f_true, ax=axs[i][j])
                cb.ax.tick_params(labelsize=15)
                for l in cb.ax.yaxis.get_ticklabels():
                    l.set_family('Times New Roman')
                tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
                cb.locator = tick_locator
                cb.update_ticks()
                if j < 2:
                    f_true.set_clim(fmin[i], fmax[i])
                    cb.ax.set_title(name_channel[i], fontdict=self.font_EN, loc='center')
                else:
                    f_true.set_clim(-limit, limit)
                    cb.ax.set_title('$\mathrm{\Delta}$' + name_channel[i], fontdict=self.font_EN, loc='center')
                # 设置刻度间隔
                axs[i][j].set_aspect(1)
                # axs[i][j].xaxis.set_major_locator(MultipleLocator(0.1))
                # axs[i][j].yaxis.set_major_locator(MultipleLocator(0.1))
                # axs[i][j].xaxis.set_minor_locator(MultipleLocator(0.2))
                # axs[i][j].yaxis.set_minor_locator(MultipleLocator(0.1))
                axs[i][j].set_xlabel(r'$x$', fontdict=self.font_EN)
                axs[i][j].set_ylabel(r'$y$', fontdict=self.font_EN)
                axs[i][j].spines['bottom'].set_linewidth(self.box_line_width)  # 设置底部坐标轴的粗细
                axs[i][j].spines['left'].set_linewidth(self.box_line_width)  # 设置左边坐标轴的粗细
                axs[i][j].spines['right'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细
                axs[i][j].spines['top'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细

    def plot_fields_ms(self, fig, axs, real, pred, coord, cmin_max=None, fmin_max=None, show_channel=None):
        if len(axs.shape) == 1:
            axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(len(self.field_name))

        name_channel = [self.field_name[i] for i in show_channel]

        if fmin_max == None:
            fmin, fmax = real.min(axis=(0, 1)), real.max(axis=(0, 1))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max == None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        x_pos = coord[:, :, 0]
        y_pos = coord[:, :, 1]
        num_channel = len(show_channel)

        # plt.clf()
        titles = ['truth', 'predicted', 'error']
        cmaps = ['RdYlBu_r', 'RdYlBu_r', 'coolwarm']
        for i in range(num_channel):

            fi = show_channel[i]
            ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
            limit = max(abs(ff[-1].min()), abs(ff[-1].max()))
            for j in range(3):

                axs[i][j].cla()
                f_true = axs[i][j].pcolormesh(x_pos, y_pos, ff[j], cmap=cmaps[j], shading='gouraud',
                                              antialiased=True, snap=True)
                f_true.set_zorder(10)
                axs[i][j].axis([cmin[0], cmax[0], cmin[1], cmax[1]])
                # ax[i][j].grid(zorder=0, which='both', color='grey', linewidth=1)
                axs[i][j].set_title(titles[j], fontdict=self.font_EN)
                # if i == 0:
                #     ax[i][j].set_title(titles[j], fontdict=self.font_CHN)
                cb = fig.colorbar(f_true, ax=axs[i][j])
                cb.ax.tick_params(labelsize=self.font['size'])
                for l in cb.ax.yaxis.get_ticklabels():
                    l.set_family('Times New Roman')
                tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
                cb.locator = tick_locator
                cb.update_ticks()
                if j < 2:
                    f_true.set_clim(fmin[i], fmax[i])
                    cb.ax.set_title(name_channel[i], fontdict=self.font_EN, loc='center')
                else:
                    f_true.set_clim(-limit, limit)
                    cb.ax.set_title('$\mathrm{\Delta}$' + name_channel[i], fontdict=self.font_EN, loc='center')
                # 设置刻度间隔
                axs[i][j].set_aspect(1)
                axs[i][j].set_xlabel(r'$x$/m', fontdict=self.font_EN)
                axs[i][j].set_ylabel(r'$y$/m', fontdict=self.font_EN)
                axs[i][j].spines['bottom'].set_linewidth(self.box_line_width)  # 设置底部坐标轴的粗细
                axs[i][j].spines['left'].set_linewidth(self.box_line_width)  # 设置左边坐标轴的粗细
                axs[i][j].spines['right'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细
                axs[i][j].spines['top'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细

    def plot_fields_am(self, fig, axs, out_true, out_pred, coord, p_id, ):

        fmax = out_true.max(axis=(0, 1, 2))  # 云图标尺
        fmin = out_true.min(axis=(0, 1, 2))  # 云图标尺

        def anim_update(t_id):
            # print('para:   ' + str(p_id) + ',   time:   ' + str(t_id))
            axes = self.plot_fields_ms(fig, axs, out_true[t_id], out_pred[t_id], coord, fmin_max=(fmin, fmax))
            return axes

        anim = FuncAnimation(fig, anim_update,
                             frames=np.arange(0, out_true.shape[0]).astype(np.int64), interval=200)

        anim.save(os.path.join(self.log_dir, str(p_id) + ".gif"), writer='pillow', dpi=300)

    def plot_box(self, fig, ax, data, title=None, legends=None, xlabel=None, xticks=None, bag_width=1.0):
        ax.set_title(title)
        ax.semilogy()
        ax.grid()
        n_vin = data.shape[-1]
        colors_map = ['#E4DACE', '#E5BB4B', '#498EAF', '#631F16']
        if len(data.shape) == 2:
            positions = np.arange(n_vin) + 1
            x_pos = None
            n_bag = 1
        else:
            n_bag = data.shape[-2]
            p = (np.linspace(0, 1, n_vin + 2) - 0.5) * bag_width
            positions = np.hstack([p[1:-1] + 0.5 + i for i in range(n_bag)]) * n_vin
            x_pos = np.arange(n_bag) * n_vin + n_vin / 2

        parts = ax.boxplot(data.reshape(data.shape[0], -1), widths=0.5 * bag_width, positions=positions, vert=True,
                           patch_artist=True, )

        for i in range(n_vin):
            for j in range(n_bag):
                parts['boxes'][i + j * n_vin].set_facecolor(colors_map[i])  # violin color
                parts['boxes'][i + j * n_vin].set_edgecolor('grey')  # violin edge
                parts['boxes'][i + j * n_vin].set_alpha(0.9)
        ax.legend(legends)
        if xticks is None:
            xticks = np.arange(n_vin * n_bag)
        ax.set_xlabel(xlabel)
        ax.set_axis_style(ax, xticks, x_pos)

    def plot_violin(self, fig, ax, data, title=None, legends=None, xticks=None, xlabel=None, bag_width=1.0):
        ax.set_title(title)
        ax.semilogy()
        ax.grid()
        n_vin = data.shape[-1]
        colors_map = ['#E4DACE', '#E5BB4B', '#498EAF', '#631F16']
        if len(data.shape) == 2:
            positions = np.arange(n_vin) + 1
            x_pos = None
            n_bag = 1
        else:
            n_bag = data.shape[-2]
            p = (np.linspace(0, 1, n_vin + 2) - 0.5) * bag_width
            positions = np.hstack([p[1:-1] + 0.5 + i for i in range(n_bag)]) * n_vin
            x_pos = np.arange(n_bag) * n_vin + n_vin / 2

        parts = ax.violinplot(data.reshape(data.shape[0], -1), widths=0.5 * bag_width, positions=positions,
                              showmeans=False, showmedians=False, showextrema=False)

        for i in range(n_vin):
            for j in range(n_bag):
                parts['bodies'][i + j * n_vin].set_facecolor(colors_map[i])  # violin color
                parts['bodies'][i + j * n_vin].set_edgecolor('grey')  # violin edge
                parts['bodies'][i + j * n_vin].set_alpha(0.9)
        ax.legend(legends)
        quartile1, medians, quartile3 = np.percentile(data.reshape(data.shape[0], -1), [25, 50, 75], axis=0)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data.reshape(data.shape[0], -1), quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        ax.scatter(positions, medians, marker='o', color='white', s=5, zorder=3)
        ax.vlines(positions, quartile1, quartile3, color='black', linestyle='-', lw=5)
        # ax.vlines(positions, whiskers_min, whiskers_max, color='black', linestyle='-', lw=1)
        if xticks is None:
            xticks = np.arange(n_vin * n_bag)
        ax.set_xlabel(xlabel)
        ax.set_axis_style(ax, xticks, x_pos)
