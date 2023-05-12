import numpy as np
import yaml
from Utilizes.process_data import DataNormer, MatLoader
from post_process.post_data import Post_2d
import os

def get_grid(real_path = None):
    xx = np.linspace(-0.127, 0.126, 64)
    xx = np.tile(xx, [64,1])

    if real_path is None:
        hub_file = os.path.join('data', 'hub_lower.txt')
        shroud_files = os.path.join('data', 'shroud_upper.txt')
    else:
        hub_file = os.path.join(real_path, 'hub_lower.txt')
        shroud_files = os.path.join(real_path, 'shroud_upper.txt')

    hub = np.loadtxt(hub_file)
    shroud = np.loadtxt(shroud_files)

    yy = []
    for i in range(64):
        yy.append(np.linspace(hub[i],shroud[i],64))

    yy = np.concatenate(yy, axis=0)
    yy = yy.reshape(64, 64).T
    xx = xx.reshape(64, 64)

    return np.concatenate([xx[:,:,np.newaxis],yy[:,:,np.newaxis]],axis=2)

def get_origin_old():
    # sample_num = 500
    # sample_start = 0

    design_files = [os.path.join('data', 'rotor37_600_sam.dat'),
                    os.path.join('data', 'rotor37_900_sam.dat')]
    field_paths = [os.path.join('data', 'Rotor37_span_600_data_64cut_clean'),
                   os.path.join('data', 'Rotor37_span_900_data_64cut_clean')]

    fields = []
    case_index = []
    for path in field_paths:
        names = os.listdir(path)
        fields.append([])
        case_index.append([])
        for i in range(len(names)):
            # 处理后数据格式为<class 'tuple'>: (3, 5, 64，64)
            if 'case_' + str(i) + '.npy' in names:
                fields[-1].append(np.load(os.path.join(path, 'case_' + str(i) + '.npy'))
                                  .astype(np.float32).transpose((1, 2, 0, 3)))
                case_index[-1].append(i)
        fields[-1] = np.stack(fields[-1], axis=0)

    design = []
    for i, file in enumerate(design_files):
        design.append(np.loadtxt(file, dtype=np.float32)[case_index[i]])

    design = np.concatenate(design, axis=0)
    fields = np.concatenate(fields, axis=0)

    return design, fields

def get_origin_6field(realpath=None):
    if realpath is None:
        sample_files = [os.path.join("data","sampleRst_500"),
                       os.path.join("data", "sampleRst_900"),
                        os.path.join("data", "sampleRst_1500")]
    else:
        sample_files = [os.path.join(realpath,"sampleRst_500"),
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

def get_origin(quanlityList=None,
                realpath=None,
                existcheck=True,
                shuffled=False,
                getridbad=True):

    if quanlityList is None:
        quanlityList = ["Static Pressure", "Static Temperature",
                        'V2', 'W2', "DensityFlow"]
    if realpath is None:
        sample_files = [os.path.join("data", "sampleRstZip_1500"),
                        os.path.join("data", "sampleRstZip_500"),
                        os.path.join("data", "sampleRstZip_970")
                        ]

    else:
        sample_files = [os.path.join(realpath, "sampleRstZip_1500"),
                        os.path.join(realpath, "sampleRstZip_500"),
                        os.path.join(realpath, "sampleRstZip_970")
                        ]
    if existcheck:
        sample_files_exists = []
        for file in sample_files:
            if os.path.exists(file + '.mat'):
                sample_files_exists.append(file)
            else:
                print("The data file {} is not exist, CHECK PLEASE!".format(file))

        sample_files = sample_files_exists


    design = []
    fields = []
    for ii, file in enumerate(sample_files):
        reader = MatLoader(file, to_torch=False)
        design.append(reader.read_field('design'))
        output = np.zeros([design[ii].shape[0], 64, 64, len(quanlityList)])
        Cp = 1004
        for jj, quanlity in enumerate(quanlityList):
            if quanlity=="DensityFlow": #设置一个需要计算获得的数据
                Vm = np.sqrt(np.power(reader.read_field("Vxyz_X"), 2) + np.power(reader.read_field("Vxyz_Y"), 2))
                output[:, :, :, jj] = (reader.read_field("Density") * Vm).copy()
            elif quanlity == "W2": #设置一个需要计算获得的数据
                output[:, :, :, jj] = 2 * Cp * (reader.read_field("Relative Total Temperature") - reader.read_field("Static Temperature")).copy()
            elif quanlity == "V2":  # 设置一个需要计算获得的数据
                output[:, :, :, jj] = 2 * Cp * (reader.read_field("Absolute Total Temperature") - reader.read_field("Static Temperature")).copy()
            else:
                output[:, :, :, jj] = reader.read_field(quanlity).copy()
        fields.append(output)

    design = np.concatenate(design, axis=0)
    fields = np.concatenate(fields, axis=0)

    if getridbad:
        if realpath is None:
            file_path = os.path.join("data", "sus_bad_data.yml")
        else:
            file_path = os.path.join(realpath, "sus_bad_data.yml")
        with open(file_path, 'r') as f:
            sus_bad_dict = yaml.load(f, Loader=yaml.FullLoader)
        sus_bad_idx = []
        for key in sus_bad_dict.keys():
            sus_bad_idx.extend(sus_bad_dict[key])
        sus_bad_idx = np.array(sus_bad_idx)
        sus_bad_idx = np.unique(sus_bad_idx)

        design = np.delete(design, sus_bad_idx, axis=0)
        fields = np.delete(fields, sus_bad_idx, axis=0)

    if shuffled:
        np.random.seed(8905)
        idx = np.random.permutation(design.shape[0])
        # print(idx[:10])
        design = design[idx]
        fields = fields[idx]

    return design, fields



def get_value(data_2d, input_para=None, parameterList=None):
    if not isinstance(parameterList, list):
        parameterList = [parameterList]

    if input_para is None:
        input_para = {
            "PressureStatic": 0,
            "TemperatureStatic": 1,
            "DensityFlow": 2,
            "PressureTotalW": 3,
            "TemperatureTotalW": 4,
        }

    grid = get_grid()
    post_pred = Post_2d(data_2d, grid,
                        inputDict=input_para,
                        )

    Rst = []
    for parameter_Name in parameterList:
        value = getattr(post_pred, parameter_Name)
        value = post_pred.span_density_average(value[..., -1])
        Rst.append(value)

    return np.concatenate(Rst, axis=1)

if __name__ == "__main__":
    design, field = get_origin()
    grid = get_grid()
    Rst = get_value(field, parameterList="EntropyStatic")
    # np.savetxt(os.path.join("Rst.txt"), Rst)
    # file_path = os.path.join("data", "sus_bad_data.yml")
    # import yaml
    # with open(file_path,'r') as f:
    #     data = yaml.load(f, Loader=yaml.FullLoader)