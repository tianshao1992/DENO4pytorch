import os
import shutil
import sys
sys.path.append(r"E:\WQN\EngTestTool\NumecaScript/")
from Auto_Numeca import makeSTR, runIggScript, runFineScript, runSolver
from Numeca_Project import NumecaPrj
from Read_Numeca import readmf

pathScript = r"E:\WQN\EngTestTool\NumecaScript\NumecaPyTemplate/"
pathPyScript = r"E:\WQN\EngTestTool\NumecaScript/"
#建立工程类
pathPrj = r"G:\WQN\anto_off_line\rotor37_case\stage_off_line_5/"
PrjCur = NumecaPrj(pathPrj)

#设置参数
Spre_design = 115000 #设计背压
Spre_range = [100000,180000] #背压范围
Spre_step = 5000 #初始背压跨度
Spre_step_MIN = 50
step_ratio = 0.7

Spre_list = [Spre_design]
while Spre_list[0]>=Spre_range[0]:
        Spre_list = [Spre_list[0] - Spre_step] + Spre_list

curIndex = Spre_list.index(Spre_design)
computation_list = [x for x in range(1,len(Spre_list))] + [0]
#修改参考设计名称
makeSTR(pathScript + "set_computations.tpl",
    pathPyScript + "fine.py",
    iec_load=PrjCur.iecfile,
    computation=0, #当前只有一个computation
    new = "NONEW",
    newname= "computation_" + str(Spre_design),
    blockid = 0,
    # bounarycode="[27,40]",  # 表示出口静压
    # spre=138888,
    itermax=300)
runFineScript(PrjCur.pathFine, pathPyScript + "fine.py")
#计算堵塞工况--不需要初始流场--立即开始并行计算
for ii in range(curIndex):
    makeSTR(pathScript + "set_computations.tpl",
            pathPyScript + "fine.py",
            iec_load=PrjCur.iecfile,
            computation=ii+1,  # 当前只有一个computation
            initial = "turbo",
            new="",
            newname="computation_" + str(Spre_list[ii]),
            blockid=0,
            bounarycode="[27,40]",  # 表示出口静压
            spre=Spre_list[ii],
            itermax=300)
    runFineScript(PrjCur.pathFine, pathPyScript + "fine.py")
    # 直接开始计算
    # runSolver(PrjCur.pathSolver, PrjCur.runfile("computation_" + str(Spre_list[ii])), "nowait")

#计算失速工况--需要初始流场--串行计算
curIndex = curIndex + 1
newIndex = ""
while Spre_step>=Spre_step_MIN:
    #分析本次计算的背压与序号
    ii = curIndex
    curSpre = Spre_list[ii-1] + Spre_step
    #生成并计算
    makeSTR(pathScript + "set_computations.tpl",
            pathPyScript + "fine.py",
            iec_load=PrjCur.iecfile,
            computation=ii,  # 当前只有一个computation
            initial="file",
            refile = PrjCur.runfile("computation_" + str(Spre_list[ii-1])),
            new = newIndex,
            # 初始流场是背压更小的前一个工况
            newname="computation_" + str(curSpre),
            blockid=0,
            bounarycode="[27,40]",  # 表示出口静压
            spre=curSpre,
            itermax=300)
    runFineScript(PrjCur.pathFine, pathPyScript + "fine.py")
    # 直接开始计算
    runSolver(PrjCur.pathSolver, PrjCur.runfile("computation_" + str(curSpre)), "wait")
    #判断是否成功计算：
    mfdata = readmf(PrjCur.mffile("computation_" + str(curSpre)),'Mass_flow')
    if abs((mfdata[0]-mfdata[1])/mfdata[0])<1e-2:
        #将刚刚计算结果记录下来
        Spre_list.append(curSpre)
        computation_list.append(ii)

        curIndex = curIndex+1
        newIndex = "" # 下次是一个新的计算
    else:
        # 更新范围
        Spre_step = Spre_step * step_ratio
        Spre_step = int(round(Spre_step/50,0)*50)
        newIndex = "NONEW"  # 下次并不进行新的计算



