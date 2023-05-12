import os
import shutil
import sys
import numpy as np
sys.path.append(r"D:\WQN\CODE\EngTestTool\NumecaScript/")
from Auto_Numeca import makeSTR, runIggScript, runFineScript, runSolver, runCFview
from Numeca_Project import NumecaPrj
from scipy.interpolate import interp1d



def readOnePrj(pathPrj,pathSave,positionZlist,
               quanlitylist=["Static Pressure","Static Temperature","Density","Vxyz_X","Vxyz_Y","Vxyz_Z"]):
        isCreated = os.path.exists(pathSave)
        if not isCreated: os.mkdir(pathSave)

        PrjCur = NumecaPrj(pathPrj)
        for name in ["computation_1"]:
                runfile = PrjCur.runfile(name)

                pathScript = os.path.join("NumecaPyTemplate")
                pathPyScript = pathPrj


                makeSTR( os.path.join(pathScript,"get_span.tpl"),
                        os.path.join(pathPyScript, "span.py"),
                        run_load = runfile,
                        upper = 0.27,
                        lower = 0.15,
                        quanlitylist=quanlitylist,
                        positionZlist = positionZlist,
                        span_save = pathSave +"/"+ name)
                runCFview(PrjCur.pathCFview, os.path.join(pathPyScript, "span.py"),runtype="nowait",safetime=15)

def readCfviewRstAll(cfviewRst,
                  positionZlist = [],
                  computationlist = ["computation_1"],
                  quanlitylist = ["Static Pressure","Static Temperature","Density","Vxyz_X","Vxyz_Y","Vxyz_Z"]):
        samdata = np.empty([len(computationlist),64,64,len(quanlitylist)], dtype=float)
        for jj, computation in enumerate(computationlist):
            for kk, quanlity in enumerate(quanlitylist):
                for ll, positionZ in enumerate(positionZlist):
                    filename = os.path.join(cfviewRst,computation + "_" + quanlity + "_" + str(positionZ) + '.dat')
                    # print(filename)

                    data = read_txt_to_ndarray(filename)
                    x = data[:, 1]
                    y = data[:, 3]
                    # 对原有曲线进行插值
                    new_x = getx(ll)
                    f = interp1d(x, y, kind='linear')
                    new_y = f(new_x)
                    samdata[jj, ll, :, kk] = new_y.copy()
        # 绘制原有曲线和插值后的曲线
        return samdata


def readCfviewRst(cfviewRst,
                  positionZlist=[],
                  computationlist=["computation_1"],
                  quanlitylist=["Static Pressure", "Static Temperature", "Density", "Vxyz_X", "Vxyz_Y", "Vxyz_Z"]
                  ):
    samdata = np.empty([len(computationlist), 64, 64, len(quanlitylist)], dtype=float)
    for jj, computation in enumerate(computationlist):
        for kk, quanlity in enumerate(quanlitylist):
            for ll, positionZ in enumerate(positionZlist):
                filename = os.path.join(cfviewRst, computation + "_" + quanlity + "_" + str(positionZ) + '.dat')
                # print(filename)

                data = read_txt_to_ndarray(filename)
                x = data[:, 1]
                y = data[:, 3]
                # 对原有曲线进行插值
                new_x = getx(ll)
                f = interp1d(x, y, kind='linear')
                new_y = f(new_x)
                samdata[jj, :, ll, kk] = new_y.copy()

    # 整理结果到dict
    samdict = {}
    for kk, quanlity in enumerate(quanlitylist):
        samdict.update({quanlity: samdata[:, :, :, kk]})

    return samdict
def getx(ll):
    pathUpper = os.path.join("data","shroud_upper.txt")
    pathLower = os.path.join("data","hub_lower.txt")

    data_upper = np.loadtxt(pathUpper, skiprows=0)
    data_lower = np.loadtxt(pathLower, skiprows=0)
    return np.linspace(data_lower[ll], data_upper[ll], 64)

def read_txt_to_ndarray(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if line.strip() and not line.startswith('#'):
                try:
                    data.append([float(val) for val in line.split()])
                except ValueError:
                    pass
    return np.array(data)

if __name__ == "__main__":
        #definite
        temp = np.linspace(-0.127, 0.126, 64).tolist()
        temp = [float(format(x, '.3f')) for x in temp]

        for ii in range(1000):
            print(ii)
            pathCase = os.path.join(r"G:\WQN\Rotor37_span\StageList_1000/","case_"+str(ii))
            pathPrj = os.path.join(pathCase,"stage")
            pathSave = os.path.join(pathCase,"cfviewRst")
            readOnePrj(pathPrj, pathSave,temp,
                       quanlitylist=['Relative Total Pressure','Relative Total Temperature','Entropy'])

            # samdata = readCfviewRst(pathSave,positionZlist=temp)
            # np.save(os.path.join(pathCase,'sampleRst.npy'), samdata)

            samdict = readCfviewRst(pathSave,
                                    positionZlist=temp,
                                    quanlitylist = ["Static Pressure","Static Temperature","Density",
                                                    "Vxyz_X","Vxyz_Y","Vxyz_Z",
                                                    'Relative Total Pressure','Relative Total Temperature',
                                                    'Absolute Total Pressure', 'Absolute Total Temperature',
                                                    'Entropy']
                                    )
            np.savez(os.path.join(pathCase,'sampleRstZip.npz'), **samdict)


