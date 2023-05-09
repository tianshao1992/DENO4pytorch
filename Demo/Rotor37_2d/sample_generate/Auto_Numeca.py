import re
import os
import shutil
import subprocess
import time

def makeSTR(pathTemplate,ScriptNew,**kwargs):
    for key in kwargs.keys():
        if isinstance(kwargs[key],str):
            kwargs[key] = kwargs[key].replace("\\","/")
    if "level" in kwargs.keys():
        if isinstance(kwargs["level"],list):
            kwargs["num"] = len(kwargs["level"])
            kwargs["level"] = [str(x) for x in kwargs["level"]]
            kwargs["level"] = "[" + str(",".join(kwargs["level"])) + "]"
    if "new" in kwargs.keys():
        if kwargs["new"] == "NONEW":
            kwargs.pop("new")


    with open(pathTemplate, 'r', encoding='utf-8') as f:
        PyTemplate = f.readlines()
    PyScript = []
    pattern = r"\*{3}(\w+)\*{3}"
    for line in PyTemplate:
        match = re.findall(pattern, line)
        if match:
            for ii in range(len(match)):
                if match[ii] in kwargs.keys():
                    symbol = "***" + match[ii] + "***"
                    line = line.replace(symbol, str(kwargs[match[ii]]))
                else:
                    line = "#" + line
                    print("The parameter {} is not input !!! CHECK PLEASE !!!".format(match[ii]))
        PyScript.append(line)

    with open(ScriptNew, 'w', encoding='utf-8') as f:
        f.writelines(PyScript)

def runIggScript(pathIGG, pathPyScript,runtype="wait", safetime=5):
    arguement = pathIGG + " -autogrid5 -batch -script " + '\"' + pathPyScript + '\"'
    if runtype == "wait":
        os.system(arguement)
    else:
        subprocess.Popen(arguement)
        time.sleep(safetime)

def runFineScript(pathFine, pathPyScript):
    arguement = pathFine + " -batch -script " + '\"' + pathPyScript + '\"'
    os.system(arguement)

def runSolver(pathSolver, runfile, runtype="nowait", safetime=5):
    arguement = pathSolver + " " +runfile.replace("\\","/")
    if runtype=="wait":
        os.system(arguement)
    else:
        subprocess.Popen(arguement)
        time.sleep(safetime)

def runCFview(pathCFview, pathPyScript, runtype="nowait", safetime=20):
    arguement = pathCFview + " -batch -macro " + '\"' + pathPyScript + '\"'
    if runtype=="wait":
        os.system(arguement)
    else:
        subprocess.Popen(arguement)
        time.sleep(safetime)


if __name__ == "__main__":
    import sys

    sys.path.append(r"E:\WQN\EngTestTool\NumecaScript/")
    from Numeca_Project import NumecaPrj


    pathScript = r"E:\WQN\EngTestTool\NumecaScript\NumecaPyTemplate/"
    pathPyScript = r"E:\WQN\EngTestTool\NumecaScript/"
    # 建立工程类
    pathPrj = r"G:\WQN\anto_off_line\rotor37_case\stage_off_line_3/"
    PrjCur = NumecaPrj(pathPrj)

    curIndex = 8
    newIndex = "NONEW"
    ii = curIndex
    curSpre = 140000
    #生成并计算
    # makeSTR(pathScript + "set_computations.tpl",
    #         pathPyScript + "fine.py",
    #         iec_load=PrjCur.iecfile,
    #         computation=ii,  # 当前只有一个computation
    #         initial="file",
    #         refile = PrjCur.runfile("computation_" + str(135000)),
    #         new = newIndex,
    #         # 初始流场是背压更小的前一个工况
    #         newname="computation_" + str(curSpre),
    #         bounarycode="[27,40]",  # 表示出口静压
    #         spre=curSpre,
    #         itermax=150)
    runFineScript(PrjCur.pathFine, pathPyScript + "fine.py")

