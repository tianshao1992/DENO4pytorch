import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sbn
import pandas as pd
from scipy import stats


class matplotlib_vision(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 40}

    def plot_loss(self, x, y, label, title=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        plt.plot(x, y, label=label)
        plt.semilogy()
        plt.grid(True)  # 添加网格
        plt.legend(loc="upper right", prop=self.font)
        plt.xlabel('iterations', self.font)
        plt.ylabel('loss value', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)
        # plt.pause(0.001)



    def plot_fields(self, x, y, real, pred, xmin_max=None, show_channel=(0,), name_channel=('T')):

        plt.rcParams['font.size'] = self.font['size']
        plt.rcParams['font.family'] = 'Times New Roman'

        fmin, fmax = real.min(axis=(0,)), real.max(axis=(0,))
        xmin,xmax=x.min(axis=(0,)),x.max(axis=(0,))
        ymin, ymax = y.min(axis=(0,)), y.max(axis=(0,))
        if xmin_max == None:
            x_lower, x_upper = xmin, xmax
            y_lower, y_upper = ymin, ymax
        else:
            x_lower, x_upper = -0.2, 1.2
            y_lower, y_upper = -0.4, 0.4

        num_channel = len(show_channel)

        for j in range(num_channel):
            c = show_channel[j]
            plt.subplot(num_channel, 3, 1 + j * 3)
            plt.axis((x_lower, x_upper, y_lower, y_upper))
            # plt.title("field " + name_channel[j] + " real", fontsize=self.font['size'])
            plt.scatter(x, y, c=real[:, c], vmin=min(real[:, c]), vmax=max(real[:, c]),
                        cmap="RdYlBu_r", marker='*', s=10.0, linewidths=5)
            # plt.axis('equal')
            plt.clim(vmin=fmin[c], vmax=fmax[c])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=self.font['size'])  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = self.font['size']

            plt.subplot(num_channel, 3, 2 + j * 3)
            plt.axis((x_lower, x_upper, y_lower, y_upper))
            # plt.title("field " + name_channel[j] + " pred", fontsize=self.font['size'])
            plt.scatter(x, y, c=pred[:, c], vmin=min(real[:, c]), vmax=max(real[:, c]),
                        cmap="RdYlBu_r", marker='*', s=10.0, linewidths=5)
            # plt.axis('equal')
            plt.clim(vmin=fmin[c], vmax=fmax[c])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=self.font['size'])  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = self.font['size']

            err = pred[:, c] - real[:, c]
            plt.subplot(num_channel, 3, 3 + j * 3)
            plt.axis((x_lower, x_upper, y_lower, y_upper))
            # plt.title("field " + name_channel[j] + " error", fontsize=self.font['size'])
            plt.scatter(x, y, c=pred[:, c] - real[:, c],
                        cmap="coolwarm", marker='*', s=10.0, linewidths=5)
#            plt.axis('equal')
 #           limit = max(abs(err.min()), abs(err.max()))
 #           plt.clim(vmin=-limit, vmax=limit)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=self.font['size'])  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = self.font['size']

    def plot_fields_ASC(self, index, x, y, real, pred, elemt):

        names = ('Uy/mm', 'Ux/mm', 'Syy/MPa', 'Sxx/MPa', 'Sxy/MPa', 'VonMises/MPa',
                 'Eyy', 'Exx', 'Exy')
        names_ = ('Uy_/mm', 'Ux_/mm', 'Syy_/MPa', 'Sxx_/MPa', 'Sxy_/MPa', 'VonMises_/MPa',
                  'Eyy_', 'Exx_', 'Exy_')
        names = ('p/Pa', 't/K', 'V/ms-1', 'vu/ms-1', 'vv/ms-1',
                  'vw/ms-1')
        names2 = ('p2/Pa', 't2/K', 'V2/ms-1', 'vu2/ms-1', 'vv2/ms-1',
                  'vw2/ms-1')
        names_ = ('p_/Pa', 't_/K', 'V_/ms-1', 'vu_/ms-1', 'vv_/ms-1',
                  'vw_/ms-1')

        x_ = x
        y_ = y
        coord = np.stack((x_, y_), axis=-1)
        err = (real - pred)
        output = np.concatenate((coord, real, pred, err), axis=-1)

        d1 = pd.DataFrame(output)
        d2 = pd.DataFrame(elemt)
        d2 = d2 + 1
        # num_nodes=coord_fields.shape[0]
        # num_elements=cell.shape[0]
        output_file = self.log_dir + '\\' + str(index) + '.dat'

        f = open(output_file, "w")

        f.write("%s\n" % ('TITLE = "Element Data"'))
        f.write("%s" % ('VARIABLES = "X/m","Y/m",'))

        for i in range(len(names)):
            f.write("%s" % ('"' + names[i] + '",'))

        for i in range(len(names)):
            f.write("%s" % ('"' + names2[i] + '",'))

        for i in range(len(names)-1):
            f.write("%s" % ('"' + names_[i] + '",'))

        f.write("%s\n" % ('"' + names_[-1] + '"'))

        f.write("%s" % ('ZONE T="Turbo surface1", '))
        f.write("%s\n" % ('Nodes=' + str(coord.shape[0]) + ', Elements=' + str(elemt.shape[0])
                          + ', F=FEPOINT, ET=QUADRILATERAL'))

        f.close()

        d1.to_csv(output_file, index=False, mode='a', float_format="%15.5e", sep=",", header=False)
        d2.to_csv(output_file, index=False, mode='a', float_format="10d", sep=" ", header=False)



    def plot_fields_ASC2(self, index, x, y, z, real, pred, elemt):


        names = ('p/Pa', 't/K', 'V/ms-1', 'vu/ms-1', 'vv/ms-1',
                  'vw/ms-1')
        names2 = ('p2/Pa', 't2/K', 'V2/ms-1', 'vu2/ms-1', 'vv2/ms-1',
                  'vw2/ms-1')
        names_ = ('p_/Pa', 't_/K', 'V_/ms-1', 'vu_/ms-1', 'vv_/ms-1',
                  'vw_/ms-1')

        x_ = x
        y_ = y
        z = z
        coord = np.stack((x_, y_, z), axis=-1)
        err = real - pred
        output = np.concatenate((coord, real, pred, err), axis=-1)

        d1 = pd.DataFrame(output)
        d2 = pd.DataFrame(elemt)
        d2 = d2 + 1

        # num_nodes=coord_fields.shape[0]
        # num_elements=cell.shape[0]
        output_file = self.log_dir + '\\' + str(index) + '.dat'

        f = open(output_file, "w")

        f.write("%s\n" % ('TITLE = "Element Data"'))
        f.write("%s" % ('VARIABLES = "X/m","Y/m","Z/m",'))

        for i in range(len(names)):
            f.write("%s" % ('"' + names[i] + '",'))

        for i in range(len(names)):
            f.write("%s" % ('"' + names2[i] + '",'))

        for i in range(len(names)-1):
            f.write("%s" % ('"' + names_[i] + '",'))

        f.write("%s\n" % ('"' + names_[-1] + '"'))

        f.write("%s" % ('ZONE T="Turbo surface1", '))
        f.write("%s\n" % ('Nodes=' + str(coord.shape[0]) + ', Elements=' + str(elemt.shape[0])
                          + ', F=FEPOINT, ET=QUADRILATERAL'))

        f.close()

        d1.to_csv(output_file, index=False, mode='a', float_format="%15.5e", sep=",", header=False)
        d2.to_csv(output_file, index=False, mode='a', float_format="10d", sep=" ", header=False)

    def plot_fields_ASC3(self, index, x, y, z, real, pred, elemt):

        # names = ('p/Pa', 't/K', 'V/ms-1', 'vu/ms-1', 'vv/ms-1',
        #          'vw/ms-1')
        # names2 = ('p2/Pa', 't2/K', 'V2/ms-1', 'vu2/ms-1', 'vv2/ms-1',
        #           'vw2/ms-1')
        names_ = ('t/K', 't2/K', 't_/K',)

        x_ = x
        y_ = y
        z = z
        coord = np.stack((x_, y_, z), axis=-1)
        err = real - pred
        output = np.concatenate((coord, real, pred, err), axis=-1)

        d1 = pd.DataFrame(output)
        if isinstance(elemt, tuple):
            d2 = pd.DataFrame(elemt[0])
            d2 = d2+1
            d3 = pd.DataFrame(elemt[1])
            d3 = d3+1
            d4 = pd.DataFrame(elemt[2])
            d4 = d4+1

            last_column = d4.iloc[:, -1].copy()
            d4[3] = last_column
            # num_nodes=coord_fields.shape[0]
            # num_elements=cell.shape[0]
            output_file = self.log_dir + '\\' + str(index) + '.dat'

            f = open(output_file, "w")

            f.write("%s\n" % ('TITLE = "Element Data"'))
            f.write("%s" % ('VARIABLES = "X/m","Y/m","Z/m",'))

            # for i in range(len(names_)):
            #     f.write("%s" % ('"' + names[i] + '",'))
            #
            # for i in range(len(names_)):
            #     f.write("%s" % ('"' + names2[i] + '",'))

            for i in range(len(names_) - 1):
                f.write("%s" % ('"' + names_[i] + '",'))

            f.write("%s\n" % ('"' + names_[-1] + '"'))

            f.write("%s" % ('ZONE T="Turbo surface1", '))
            f.write("%s\n" % ('Nodes=' + str(coord.shape[0]) + ', Elements=' + str(d2.shape[0]+d3.shape[0]+d4.shape[0])
                              + ', F=FEPOINT, ET=QUADRILATERAL'))
            f.close()
            d1.to_csv(output_file, index=False, mode='a', float_format="%15.5e", sep=",", header=False)
            d2.to_csv(output_file, index=False, mode='a', float_format="10d", sep=" ", header=False)
            d3.to_csv(output_file, index=False, mode='a', float_format="%15.5e", sep=",", header=False)
            d4.to_csv(output_file, index=False, mode='a', float_format="10d", sep=" ", header=False)

        else:

            d2 = pd.DataFrame(elemt)
            d2 = d2 + 1

            # num_nodes=coord_fields.shape[0]
            # num_elements=cell.shape[0]
            output_file = self.log_dir + '\\' + str(index) + '.dat'

            f = open(output_file, "w")

            f.write("%s\n" % ('TITLE = "Element Data"'))
            f.write("%s" % ('VARIABLES = "X/m","Y/m","Z/m",'))

            # for i in range(len(names)):
            #     f.write("%s" % ('"' + names[i] + '",'))
            #
            # for i in range(len(names)):
            #     f.write("%s" % ('"' + names2[i] + '",'))

            for i in range(len(names_) - 1):
                f.write("%s" % ('"' + names_[i] + '",'))

            f.write("%s\n" % ('"' + names_[-1] + '"'))

            f.write("%s" % ('ZONE T="Turbo surface1", '))
            f.write("%s\n" % ('Nodes=' + str(coord.shape[0]) + ', Elements=' + str(elemt.shape[0])
                              + ', F=FEPOINT, ET=QUADRILATERAL'))

            f.close()

            d1.to_csv(output_file, index=False, mode='a', float_format="%15.5e", sep=",", header=False)
            d2.to_csv(output_file, index=False, mode='a', float_format="10d", sep=" ", header=False)

    def plot_fields_CSV(self, index, x, y, z, real, pred, err, elemt):
        # names = ( 'T [ K ]','T2 [ K ]','T_ [ K ]')
        # names2 = ( 'T2 [ K ]')
        names_ = ('T [ K ]','T2 [ K ]','T_ [ K ]')
        x_ = x
        y_ = y
        z = z
        coord = np.stack((x_, y_, z), axis=-1)
        # err = (real - pred)
        output = np.concatenate((coord, real, pred, err), axis=-1)
        d1 = pd.DataFrame(output)
        if isinstance(elemt, tuple):
            d2 = pd.DataFrame(elemt[0])
            d2 = d2
            d3 = pd.DataFrame(elemt[1])
            d3 = d3
            d4 = pd.DataFrame(elemt[2])
            d4 = d4
            output_file = self.log_dir + '\\' + str(index) + '.csv'
            f = open(output_file, "w")
            f.write("\n")
            f.write("%s\n" % ('[Name]'))
            f.write("%s\n" % ('Turbo Surface 1'))
            f.write("\n")
            f.write("%s\n" % ('[Data]'))
            f.write("%s" % ('X [ m ], Y [ m ], Z [ m ],'))

            # for i in range(len(names)-1):
            #     f.write("%s" % (names[i] + ','))

            # for i in range(len(names)):
            #     f.write("%s" % (names2[i] + ','))
            #
            for i in range(len(names_) - 1):
                f.write("%s" % (names_[i] + ','))

            f.write("%s\n" % ( names_[-1]))
            f.close()
            d1.to_csv(output_file, index=False, mode='a', float_format="%15.5e", sep=",", header=False)
            f = open(output_file, "a")
            f.write("%s\n" % ('[Faces]'))
            f.close()
            d2.to_csv(output_file, index=False, mode='a', float_format="10d", sep=",", header=False)
            d3.to_csv(output_file, index=False, mode='a', float_format="10d", sep=",", header=False)
            d4.to_csv(output_file, index=False, mode='a', float_format="10d", sep=",", header=False)

        else:
            d2 = pd.DataFrame(elemt)
            d2 = d2
            # num_nodes=coord_fields.shape[0]
            # num_elements=cell.shape[0]
            output_file = self.log_dir + '\\' + str(index) + '.csv'
            f = open(output_file, "w")

            f.write("\n")
            f.write("%s\n" % ('[Name]'))
            f.write("%s\n" % ('Turbo Surface 1'))
            f.write("\n")
            f.write("%s\n" % ('[Data]'))
            f.write("%s" % ('X [ m ], Y [ m ], Z [ m ],'))

            # for i in range(len(names)):
            #     f.write("%s" % (names[i] + ','))
            #
            # for i in range(len(names)):
            #     f.write("%s" % (names2[i] + ','))

            for i in range(len(names_) - 1):
                f.write("%s" % (names_[i] + ','))

            f.write("%s\n" % (names_[-1]))
            f.close()
            d1.to_csv(output_file, index=False, mode='a', float_format="%15.5e", sep=",", header=False)
            f = open(output_file, "a")
            f.write("%s\n" % ('[Faces]'))
            f.close()
            d2.to_csv(output_file, index=False, mode='a', float_format="10d", sep=",", header=False)



    def plot_fields_tr(self, x, y, real, pred, foils, triangles, xmin_max=None, show_channel=(0, 1, 2, 3 ,4), name_channel=('P', 'T', 'VV' , 'U', 'V')):

        fmin, fmax = real.min(axis=(0,)), real.max(axis=(0,))

        if xmin_max == None:
            x_lower, x_upper = -1, 1
            y_lower, y_upper = -1, 1
        else:
            x_lower, x_upper = -0.2, 1.2
            y_lower, y_upper = -0.4, 0.4


        plt.rcParams['font.size'] = self.font['size']

        num_channel = len(show_channel)
        for i in range(num_channel):
            fi = show_channel[i]
            # tri_refi, f_refi = refiner.refine_field(real[:, fi], subdiv=3)

            plt.subplot(num_channel, 3, 3 * i + 1)
            plt.axis((x_lower, x_upper, y_lower, y_upper))
            plt.tripcolor(x, y, real[:, fi], triangles=triangles, cmap='RdBu_r', shading='gouraud', antialiased=True, snap=True) #画云图
            plt.fill(foils[:, 0], foils[:, 1], facecolor='w')

            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=self.font['size'])  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = self.font['size']

            # tri_refi, f_refi = refiner.refine_field(real[:, fi], subdiv=3)

            plt.subplot(num_channel, 3, 3 * i + 2)
            plt.axis((x_lower, x_upper, y_lower, y_upper))
            plt.tripcolor(x, y, pred[:, fi], triangles=triangles, cmap='RdBu_r', shading='gouraud', antialiased=True, snap=True)
            plt.fill(foils[:, 0], foils[:, 1], facecolor='w')
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=self.font['size'])  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = self.font['size']

            err = pred[:, fi] - real[:, fi]
            # tri_refi, f_refi = refiner.refine_field(err, subdiv=2)
            plt.subplot(num_channel, 3, 3 * i + 3)
            plt.axis((x_lower, x_upper, y_lower, y_upper))
            plt.tripcolor(x, y, err, triangles=triangles, cmap='coolwarm', shading='gouraud', antialiased=True, snap=True)
            plt.fill(foils[:, 0], foils[:, 1], facecolor='w')

            limit = max(abs(err.min()), abs(err.max()))
            plt.clim(vmin=-limit, vmax=limit)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=30)  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = self.font['size']



    def plot_regression(self, true, pred, axis=0, title=None):
        # 所有功率预测误差与真实结果的回归直线
        sbn.set(color_codes=True)
        sbn.set_style('white')

        max_value = max(true) # math.ceil(max(true)/100)*100
        min_value = min(true) # math.floor(min(true)/100)*100
        split_value = np.linspace(min_value, max_value, 11)

        split_dict = {}
        split_label = np.zeros(len(true), np.int)
        for i in range(len(split_value)):
            split_dict[i] = str(split_value[i])
            index = true >= split_value[i]
            split_label[index] = i + 1


        plt.scatter(true, pred, marker='.')

        plt.plot([min_value, max_value], [min_value, max_value], 'r-', linewidth=5.0)
        plt.fill_between([min_value, max_value], [0.95*min_value, 0.95*max_value], [1.05*min_value, 1.05*max_value],
                          alpha=0.2, color='b')

        # plt.ylim((min_value, max_value))
        plt.xlim((min_value, max_value))
        plt.ylabel('pred value', self.font)
        plt.xlabel('real value', self.font)
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = self.font['size']
        plt.grid(True)  # 添加网格
        plt.title(title, self.font)
        # plt.ylim((-0.2, 0.2))
        # plt.pause(0.001)

    def plot_scatter(self, true, pred, axis=0, title=None):
        # sbn.set(color_codes=True)

        plt.scatter(np.arange(true.shape[0]), true, marker='*')
        plt.scatter(np.arange(true.shape[0]), pred, marker='.')

        plt.ylabel('target value', self.font)
        plt.xlabel('samples', self.font)
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.grid(True)  # 添加网格
        plt.title(title, self.font)


    def plot_error(self, error, title=None):
        # sbn.set_color_codes()
        error = pd.DataFrame(error) * 100
        sbn.distplot(error, bins=20, norm_hist=True, rug=True, fit=stats.norm, kde=False,
                     rug_kws={"color": "g"}, fit_kws={"color": "r", "lw": 3}, hist_kws={"color": "b"})
        # plt.xlim([-1, 1])
        plt.xlabel("predicted relative error / %", self.font)
        plt.ylabel('distribution density', self.font)
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.grid(True)
        # plt.legend()
        plt.title(title, self.font)
