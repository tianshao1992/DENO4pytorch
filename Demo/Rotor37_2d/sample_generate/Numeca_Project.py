import os

class NumecaPrj:
    def __init__(self, pathPrj):
        import os
        pathPrj = pathPrj.replace('\\', '/')
        if pathPrj[-1] != '/':
            self.pathPrj = pathPrj + '/'
        else:
            self.pathPrj = pathPrj

        self.namePrj = "stage"

        self.trbfile = self.pathPrj + "_mesh/"+ self.namePrj +".trb"
        self.gemofile = self.pathPrj + "_mesh/"+ self.namePrj +".geomturbo"
        self.iecfile = self.pathPrj + self.namePrj +".iec"
        # self.runfile = self.pathPrj + "stage_computation_1\stage_computation_1.run"

        self.pathNumeca = self.getPathNumeca()
        self.pathIGG = self.pathNumeca + r"iggx86_64.exe"
        self.pathFine = self.pathNumeca + r"finex86_64.exe"
        self.pathSolver = self.pathNumeca + r"euranusx86_64.exe"
        self.pathCFview = self.pathNumeca + r"cfviewx86_64.exe"


    def getPathNumeca(self):
        name = os.popen('hostname').read()
        name = name.replace('\n','')
        pathdict = {"DESKTOP-HPU6DIC": r"C:\NUMECA_SOFTWARE\fine132\bin64/",
                    "DESKTOP-NVIN80M": r"E:\NUMECA\fine132\bin64/"}
        return pathdict[name]

    def runfile(self,computationName):
        runName = self.pathPrj\
                  + self.namePrj +"_"+computationName+"/"\
                  + self.namePrj +"_"+computationName+".run"
        return runName

    def mffile(self,computationName):
        runName = self.pathPrj\
                  + self.namePrj +"_"+computationName+"/"\
                  + self.namePrj +"_"+computationName+".mf"
        return runName



if __name__ == "__main__":
    pathPrj = r"D:\TsBMDO\7Blade_stall\template_7Blade_132_2condition_Opt_T103000_G0_I1/"
    NumecaPrj(pathPrj)