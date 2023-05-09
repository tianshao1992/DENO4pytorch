import os
import shutil
import numpy as np

def findStage(pathPrj,pathTemplate,ii): #找到stage文件的位置
    prjName=  os.path.join(pathPrj,"OptRst\Gen_0\Ind_"+str(ii)+"\stage.geomturbo")
    # print(prjName)
    templateNameList = os.listdir(pathTemplate)

    if os.path.exists(prjName):
        return prjName
    else:
        for name in templateNameList:
            if "G0_I"+str(ii) in name:
                return os.path.join(pathTemplate,name,"_mesh\stage.geomturbo")




if __name__ == "__main__":
    VarTXT = "G:\WQN\Rotor37_span\StageList_1500\Rotor37_sample_1500.txt"
    data = np.loadtxt(VarTXT)

    pathSave = "G:\WQN\Rotor37_span\StageList_1500/"
    pathPrj = "E:\WQN\ROTOR37G\OptRotor37GEng_dup_3/"
    pathTemplate = "E:\WQN\ROTOR37G\Rotor37Eng_template/"
    for ii in range(1500):
        print(str(ii))
        pathStage = findStage(pathPrj, pathTemplate, ii)
        # print(pathStage)
        pathTarget = os.path.join(pathSave,"case_"+str(ii))
        isCreated = os.path.exists(pathTarget)
        if not isCreated: os.mkdir(pathTarget)
        shutil.copy(pathStage,pathTarget)


        VarSam = data[ii,:]
        np.savetxt(os.path.join(pathTarget,"variable_28D.txt"),VarSam)

