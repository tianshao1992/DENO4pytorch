import scipy.io as scio
import numpy as np
import os

def npyTOmat(sampleNum):
    input_28D = np.zeros([sampleNum,28])
    output_64_64_5D = np.zeros([sampleNum,64,64,6])
    for ii in range(sampleNum):
        pathCase = os.path.join(r"G:\WQN\Rotor37_span\StageList_1500/", "case_" + str(ii))
        pathVar = os.path.join(pathCase, "variable_28D.txt")
        pathNpy = os.path.join(pathCase, "sampleRst.npy")
        if ii!=667:
            input = np.loadtxt(pathVar,skiprows=0)
            output = np.load(pathNpy)

        input_28D[ii,:] = input.copy()
        output_64_64_5D[ii,:,:,:] = output.copy()

    scio.savemat(os.path.join(r"G:\WQN\Rotor37_span\StageList_1500/", "sampleRst_1500.mat"),
                 {"input":input_28D,"output":output_64_64_5D})

def npzTOmat(sampleNum, work_path, quanlityList = None):
    design = np.zeros([sampleNum, 28])

    if quanlityList is None:
        npzFile = os.path.join(work_path, "case_0","sampleRstZip.npz")
        loaded_dict = np.load(npzFile)
        quanlityList = loaded_dict.files

    samDict = {}
    for quanlity in quanlityList:
        samDict.update({quanlity : np.zeros([sampleNum,64,64])})


    for ii in range(sampleNum):
        pathCase = os.path.join(work_path, "case_" + str(ii))
        pathVar = os.path.join(pathCase, "variable_28D.txt")
        pathNpz = os.path.join(pathCase, "sampleRstZip.npz")

        isExist = os.path.exists(pathNpz)
        if isExist:
            input = np.loadtxt(pathVar, skiprows=0)
            output = np.load(pathNpz)
        for quanlity in quanlityList:
            design[ii, :] = input.copy()
            samDict[quanlity][ii,:,:] = output[quanlity].copy()
    samDict["design"] = design
    scio.savemat(os.path.join(work_path, "sampleRstZip_" + str(sampleNum) + ".mat"),
                 samDict)

if __name__ == "__main__":
    sampleNum = 970
    work_path = r"G:\WQN\Rotor37_span\StageList_1000/"
    npzTOmat(sampleNum, work_path, quanlityList=None)


