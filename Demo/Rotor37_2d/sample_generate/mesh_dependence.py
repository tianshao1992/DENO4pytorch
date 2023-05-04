import os
import shutil
import sys
sys.path.append(r"E:\WQN\EngTestTool\NumecaScript/")
from Auto_Numeca import makeSTR, runIggScript, runFineScript, runSolver

pathIGG = r"E:\NUMECA\fine132\bin64\iggx86_64.exe"
pathFine = r"E:\NUMECA\fine132\bin64\\finex86_64.exe"
pathSolver = r"E:\NUMECA\fine132\bin64\euranusx86_64.exe"

for ii in range(15):
    pathPrj = r"G:\WQN\griddepend\template_7Blade_132_1condition/"
    pathPrjNew = r"G:\WQN\griddepend\prjfiles\template_7Blade_132_1condition_" + str(ii) + "/"
    if not os.path.exists(pathPrjNew):
        shutil.copytree(pathPrj, pathPrjNew)

    trbfile = pathPrjNew + "_mesh\stage.trb"
    gemofile = pathPrjNew + "_mesh\stage.geomturbo"
    iecfile = pathPrjNew + "stage.iec"
    runfile = pathPrjNew + "stage_computation_1\stage_computation_1.run"

    pathScript = r"E:\WQN\EngTestTool\NumecaScript\NumecaPyTemplate/"
    pathPyScript = r"G:\WQN\griddepend\tempfile/"

    makeSTR(pathScript + "set_mesh_level_multi_blade.tpl",
            pathPyScript + "mesh.py",
            trb_load = trbfile,
            geom = gemofile,
            level = [ii-12,ii-8,ii-12,ii-8,ii-12,ii-8,ii-12],
            trb_save = trbfile)
    runIggScript(pathIGG, pathPyScript + "mesh.py")

    makeSTR(pathScript + "set_iec.tpl",
            pathPyScript + "fine.py",
            iec_load=iecfile,
            computation=0,
            # bounarycode="[27,40]",  # 表示出口静压
            # spre=138888,
            itermax=600)
    runFineScript(pathFine, pathPyScript + "fine.py")

    runSolver(pathSolver, runfile)