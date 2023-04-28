from torch.utils.data import DataLoader
# import torch
# print(torch.__version__)
import numpy as np
import os
from Utilizes.visual_data import MatplotlibVision
from Utilizes.process_data import DataNormer, MatLoader
import matplotlib.pyplot as plt
from post_data import Post_2d
import sys
sys.path.append("..")
from utilizes_rotor37 import get_grid, get_origin

def load_Npz(npzFile, quanlityList=None):
    loaded_dict = np.load(npzFile)
    if quanlityList is None:
        quanlityList = loaded_dict.files
    fields = np.zeros([64, 64, len(quanlityList)])
    for ii, quanlity in enumerate(quanlityList):
        if quanlity in loaded_dict.files:
            fields[:,:,ii] = loaded_dict[quanlity]

    return fields

def plot_span_curve(post, parameterList, save_path = None, fig_id = 0, label = None):
# 绘制单个对象曲线
    if not isinstance(parameterList,list):
        parameterList = [parameterList]

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('unset')) # 不在此处设置名称
    for parameter_Name in parameterList:
        fig, axs = plt.subplots(1, 1, figsize=(3, 6), num=1)
        value_span = getattr(post, parameter_Name)
        Visual.plot_value(fig, axs, value_span[:, -1], np.linspace(0,1,post.n_1d), label=label,
                          title=parameter_Name, xylabels=("efficiency", "span"))

        if save_path is None:
            jpg_path = os.path.join(work_path, parameter_Name + "_" + str(fig_id) + '.jpg')
        fig.savefig(jpg_path)
        plt.close(fig)

if __name__ == "__main__":

    work_path = os.path.join("..", "data_collect")
    isCreated = os.path.exists(work_path)
    if not isCreated: os.mkdir(work_path)

    grid = get_grid(os.path.join("..", "data"))
    output = load_Npz(os.path.join("..", "data", "sampleRstZip.npz"),
                      quanlityList=["Static Pressure", "Static Temperature", "Density",
                                    'Relative Total Pressure', 'Relative Total Temperature',
                                    "Vxyz_X", "Vxyz_Y",
                                    # "Vxyz_Z",
                                    # 'Entropy'
                                    ])
    # design, fields = get_origin()

    ii = 0
    post = Post_2d(output[:,:,:],grid,
                   inputDict = {
                        "PressureStatic" : 0,
                        "TemperatureStatic" : 1,
                        "Density" : 2,
                        "PressureTotalW" : 3,
                        "TemperatureTotalW" : 4,
                        "VelocityX" : 5,
                        "VelocityY" : 6,
                                }
                   )

    fig_id = 0

    parameterList = ["Efficiency", "EntropyStatic"]
    plot_span_curve(post, parameterList, save_path=None, fig_id=0, label=None)



