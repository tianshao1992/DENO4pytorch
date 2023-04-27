from torch.utils.data import DataLoader
# import torch
# print(torch.__version__)
import numpy as np
import os
from Utilizes.visual_data import MatplotlibVision
from Utilizes.process_data import DataNormer, MatLoader
import matplotlib.pyplot as plt
from post_data import Post_2d

def get_origin(realpath=None):
    if realpath is None:
        sample_files = [
                        # os.path.join("..","data","sampleRst_500"),
                        os.path.join("..","data", "sampleRst_900"),
                        # os.path.join("..","data", "sampleRst_1500")
                        ]
    else:
        sample_files = [
                        os.path.join(realpath,"sampleRst_500"),
                        os.path.join(realpath, "sampleRst_900"),
                        os.path.join(realpath, "sampleRst_1500")]

    design = []
    fields = []
    for file in sample_files:
        reader = MatLoader(file)
        design.append(reader.read_field('input'))
        fields.append(reader.read_field('output'))

    design = np.concatenate(design, axis=0)
    fields = np.concatenate(fields, axis=0)

    return design, fields

def load_Npz(npzFile, quanlityList=None):
    loaded_dict = np.load(npzFile)
    if quanlityList is None:
        quanlityList = loaded_dict.files
    fields = np.zeros([64, 64, len(quanlityList)])
    for ii, quanlity in enumerate(quanlityList):
        if quanlity in loaded_dict.files:
            fields[:,:,ii] = loaded_dict[quanlity]

    return fields


quanlitylist = ["Static Pressure", "Static Temperature", "Density",
                "Vxyz_X", "Vxyz_Y", "Vxyz_Z",
                'Relative Total Pressure', 'Relative Total Temperature', 'Entropy']


def get_grid():
    xx = np.linspace(-0.127, 0.126, 64)
    xx = np.tile(xx, [64,1])

    hub_file = os.path.join("..", 'data','hub_lower.txt')
    hub = np.loadtxt(hub_file)
    shroud_files = os.path.join("..", 'data', 'shroud_upper.txt')
    shroud = np.loadtxt(shroud_files)

    yy = []
    for i in range(64):
        yy.append(np.linspace(hub[i],shroud[i],64))

    yy = np.concatenate(yy, axis=0)
    yy = yy.reshape(64, 64).T
    xx = xx.reshape(64, 64)

    return np.concatenate([xx[:,:,np.newaxis],yy[:,:,np.newaxis]],axis=2)


if __name__ == "__main__":

    work_path = os.path.join("figure_save")
    isCreated = os.path.exists(work_path)
    if not isCreated: os.mkdir(work_path)

    grid = get_grid()
    output = load_Npz(os.path.join("..", "data", "sampleRstZip.npz"),
                      quanlityList=["Static Pressure", "Static Temperature", "Density",
                                    'Relative Total Pressure', 'Relative Total Temperature',
                                    "Vxyz_X", "Vxyz_Y",
                                    # "Vxyz_Z",
                                    # 'Entropy'
                                    ])
    # design, fields = get_origin()
    # output = fields[:, :, :, :-1].transpose((0, 2, 1, 3))


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

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('unset'))
    fig, axs = plt.subplots(1, 1, figsize=(3, 6), num=1)
    Visual.plot_value(fig, axs, post.EntropyStatic[:, -5:], np.linspace(0,1,64), label="true",
                      title="train_solution", xylabels=("efficiency", "span"))
    fig.savefig(os.path.join(work_path, 'valid_solution_ent_' + str(fig_id) + '.jpg'))
    plt.close(fig)

    # fig, axs = plt.subplots(1, 1, figsize=(10, 3), num=2)
    # Visual.plot_value(fig, axs,np.linspace(0, 1, 64), post.span_space_average(post.PressureRatioV), label="true",
    #                   title="train_solution", xylabels=("axial", "pressureLoss"))
    # fig.savefig(os.path.join(work_path, 'valid_solution_pr_axis_' + str(fig_id) + '.jpg'))
    # fig.show()
    plt.close(fig)
