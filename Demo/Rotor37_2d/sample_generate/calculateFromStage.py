import os
import shutil
# import sys
# sys.path.append(r"E:\WQN\EngTestTool\NumecaScript/")
from Auto_Numeca import makeSTR, runIggScript, runFineScript, runSolver
from Numeca_Project import NumecaPrj

def calculateOne(pathPrj,pathStage,pathTemplate): #新工程路径，stage文件路径，模板路径
    #复制模板到新工程
    isCreated = os.path.exists(pathPrj)
    if isCreated:  shutil.rmtree(pathPrj)
    # if isCreated: pathPrj=os.path.join(os.path.dirname(pathPrj),"stage_temp")#"#shutil.rmtree(pathPrj)
    shutil.copytree(pathTemplate, pathPrj)

    #更换stage文件
    pathStageNew = os.path.join(pathPrj,"_mesh","stage.geomturbo")
    isStageIn = os.path.exists(pathStageNew)
    if isStageIn: os.rmdir(pathStageNew)
    shutil.copy(pathStage,pathStageNew)
    # #
    PrjCur = NumecaPrj(pathPrj)
    pathScript = os.path.join("NumecaPyTemplate")
    pathPyScript = pathPrj
    #
    makeSTR(pathScript + "/set_mesh_level.tpl",
            pathPyScript + "/mesh.py",
            trb_load = PrjCur.trbfile,
            geom = PrjCur.gemofile,
            trb_save = PrjCur.trbfile)
    runIggScript(PrjCur.pathIGG, pathPyScript + "/mesh.py",runtype="wait",safetime=5)

    makeSTR(pathScript + "/set_iec.tpl",
            pathPyScript + "/fine.py",
            iec_load=PrjCur.iecfile,
            computation=0,
            # bounarycode="[27,40]",  # 表示出口静压
            # spre=138888,
            itermax=300)
    runFineScript(PrjCur.pathFine, pathPyScript + "/fine.py")
    runSolver(PrjCur.pathSolver, PrjCur.runfile(computationName="computation_1"),runtype="nowait",safetime=10)


if __name__ == "__main__":
    for ii in range(970,999):
        pathTemplate = r"G:\WQN\Rotor37_span\templates\stage/"
        pathPrj = os.path.join(r"G:\WQN\Rotor37_span\StageList_1000/","case_"+str(ii))
        pathStage = os.path.join(pathPrj,"stage.geomturbo")

        calculateOne(os.path.join(pathPrj,"stage"), pathStage, pathTemplate)

