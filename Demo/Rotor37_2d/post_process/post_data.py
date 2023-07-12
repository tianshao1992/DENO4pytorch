import numpy as np
import inspect
import os

class Post_2d(object):
    def __init__(self,data_2d,grid,inputDict=None): #默认输入格式为64*64*5
        self.grid = grid
        self.data_2d = data_2d

        if inputDict is None:
            self.inputDict = {
                "PressureStatic": 0,
                "TemperatureStatic": 1,
                "V2": 2,
                "W2": 3,
                "DensityFlow": 4,
            }
        else:
            self.inputDict = inputDict

        self.input_check()
        self.set_basic_const()

        self._PressureStatic= None
        self._TemperatureStatic = None
        self._Density = None
        self._DensityFlow = None

        self._VelocityX = None
        self._VelocityY = None
        self._VelocityZ = None

        self._V2 = None
        self._W2 = None

        self._AlphaV = None
        self._MagV = None

        self._Uaxis = None

        self._WelocityX = None
        self._WelocityY = None
        self._WelocityZ = None

        self._AlphaW = None
        self._MagW = None
        #相对速度总压和绝对速度总压是不同的
        self._PressureTotalV = None
        self._TemperatureTotalV = None

        self._PressureTotalW = None
        self._TemperatureTotalW = None
        self._PressureTotalRot = None
        self._TemperatureTotalRot = None
        self._TemperatureIsentropic = None

        self._PressureRatioV = None
        self._TemperatureRatioV = None

        self._PressureRatioW = None
        self._TemperatureRatioW = None

        self._Efficiency = None
        self._EfficiencyPoly = None
        self._EfficiencyR = None
        self._PressureLoss = None
        self._DFactor = None
        self._PressureLossR = None
        self._DFactorR = None
        self._EntropyStatic = None
        self._EntropyStaticNorm = None
        self._MachIsentropic = None
        self._Load = None
        self._LoadR = None


    def set_basic_const(self,
                        kappa = 1.400,
                        Cp = 1004,
                        sigma = 1.6,
                        rotateSpeed = -17188, # rpm
                        Rg = 287
                        ):
        self.kappa = kappa
        self.Cp = Cp
        self.sigma = sigma# 代表稠度
        self.rotateSpeed = rotateSpeed
        self.Rg = Rg


    def get_parameter(self,shape_index=None): #计算各种性能参数,shape表示针对哪一部分进行计算
        if shape_index==None:
            # 如果不设置，就对全局进行计算
            shape_index = [np.arange(self.n_1d),np.arange(self.n_2d)]

    def input_check(self):
        gridshape = self.grid.shape
        datashape = self.data_2d.shape

        if len(gridshape) != 3 or gridshape[2] != 2:
            print("invalid grid input!")
        if len(gridshape) != 3 and len(gridshape):
            print("invalid data input!")
        if len(datashape) == 3:
            self.data_2d = self.data_2d[None, :, :, :]
            datashape = self.data_2d.shape
            print("one sample input!")
        if len(datashape) == 4:
            self.num = datashape[0]
            print(str(self.num) + " samples input!")
        if gridshape[:2] != datashape[1:3]:
            print("dismatch data & grid input!")

        self.n_1d = self.data_2d.shape[1]
        self.n_2d = self.data_2d.shape[2]

    def field_density_average(self, parameter_Name, shape_index=None, location="outlet"):
        data = getattr(self, parameter_Name) # 整个二维空间的取值

        # shape check
        if shape_index is None:
            shape = slice(0, None)
            if location=="outlet":
                shape = slice(-1, None)
            elif location=="near_outlet":
                shape = slice(-5, None)
            elif location=="whole":
                shape = slice(None)
        else:
            shape = slice(shape_index[0], shape_index[1])

        data = data[..., shape]

        if self.DensityFlow is not None:
            density_aver = np.mean(self.DensityFlow[..., shape], axis=1, keepdims=True)
            if len(density_aver.shape)==2:
                density_aver = density_aver[..., None]
            density_norm = self.DensityFlow[..., shape]\
                           /np.tile(density_aver, (1, self.n_2d, 1))

            return np.mean(data * density_norm, axis=1)
        else:
            print("The parameter DensityFlow is not exist, CHECK PLEASE!")
            return self.span_space_average(data)


    def span_density_average(self, data, shape_index=None, location="outlet"): #输入为2维ndarry
        # data check
        if len(data.shape)<2:
            print("error input in function span_density_average.")
        if len(data.shape)==2:
            data = data[:, :, None]

        # shape check
        if shape_index is None:
            shape = slice(0, None)
            if location=="outlet":
                shape = slice(-1, None)
        else:
            shape = slice(shape_index[0], shape_index[1])

        if self.DensityFlow is not None:
            density_aver = np.mean(self.DensityFlow[..., shape], axis=1)
            density_norm = self.DensityFlow[..., shape]\
                           /np.tile(density_aver[..., None], (1, self.n_2d, 1))

            return np.mean(data * density_norm, axis=1)
        else:
            print("The parameter DensityFlow is not exist, CHECK PLEASE!")
            return self.span_space_average(data)

    def span_space_average(self, data):
        if len(data.shape)<2:
            print("error input in function span_density_average.")

        return np.mean(data, axis=1)

    def get_variable_name(var):
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            if line.strip().startswith(var + " "):
                return line.split("=")[0].strip()
        return None

    def get_para_from_input(self, parameter):
        if parameter in self.inputDict.keys():
            rst = self.data_2d[..., self.inputDict[parameter]]
        else:
            eval("rst = self._" + parameter)
        return rst

    #=================================================================================#
#各类具体参数设置

    #===============================================================#
    #基本参数获取：1.静压；2.静温；3.密度；4.速度方向；5.速度大小
    # ==============================================================#
    def set_PressureStatic(self, x):
        self.PressureStatic = x
    def set_TemperatureStatic(self, x):
        self.TemperatureStatic = x
    def set_Density(self, x):
        self.Density = x
    def set_DensityFlow(self, x):
        self._DensityFlow = x
    def set_VelocityX(self, x):
        self.VelocityX = x
    def set_VelocityY(self, x):
        self.VelocityY = x
    def set_VelocityZ(self, x):
        self.VelocityZ = x
    def set_AlphaV(self, x):
        self.AlphaV = x
    def set_V2(self, x):
        self.V2 = x


    def get_PressureStatic(self):
        if self._PressureStatic is None:
            rst = self.data_2d[..., self.inputDict["PressureStatic"]]
            self._PressureStatic = rst
            return rst
        else:
            return self._PressureStatic
    def get_TemperatureStatic(self):
        if self._TemperatureStatic is None:
            rst = self.data_2d[..., self.inputDict["TemperatureStatic"]]
            self._TemperatureStatic = rst
            return rst
        else:
            return self._TemperatureStatic
    def get_Density(self):
        if self._Density is None:
            if "Density" in self.inputDict.keys():
                return self.data_2d[..., self.inputDict["Density"]]
            else:
                print("Density" + " is not exist")
        else:
            return self._Density
    def get_DensityFlow(self):
        if self._DensityFlow is None:
            if "DensityFlow" in self.inputDict.keys():
                rst = self.data_2d[..., self.inputDict["DensityFlow"]]
            else:
                if self.VelocityX is not None:
                    rst = self.Density * self.VelocityX
                else:
                    rst = None
            self._DensityFlow = rst
            return rst
        else:
            return self._DensityFlow
    def get_VelocityX(self):
        if self._VelocityX is None:
            if "VelocityX" in self.inputDict.keys():
                rst = self.data_2d[..., self.inputDict["VelocityX"]]
            else:
                rst = None
            self._VelocityX = rst
            return rst
        else:
            return self._VelocityX
    def get_VelocityY(self):
        if self._VelocityY is None:
            rst = self.data_2d[..., self.inputDict["VelocityY"]]
            self._VelocityY = rst
            return rst
        else:
            return self._VelocityY
    def get_VelocityZ(self):
        if self._VelocityZ is None:
            rst = np.zeros([self.num, self.n_1d, self.n_2d])
            self._VelocityZ = rst
            return rst
        else:
            return self._VelocityZ

    def get_AlphaV(self):
        if self._AlphaV is None:
            return np.arctan(self.VelocityY / self.VelocityX)
        else:
            return self._AlphaV

    def get_V2(self):
        if self._V2 is None:
            if "V2" in self.inputDict.keys():
                rst = self.data_2d[..., self.inputDict["V2"]]
            else:
                rst = np.power(self.VelocityX,2) + np.power(self.VelocityY,2) + np.power(self.VelocityZ,2)
            self._V2 = rst
            return rst
        else:
            return self._MagV


    PressureStatic = property(get_PressureStatic, set_PressureStatic)
    TemperatureStatic = property(get_TemperatureStatic, set_TemperatureStatic)
    Density = property(get_Density, set_Density)
    DensityFlow = property(get_DensityFlow, set_DensityFlow)
    VelocityX = property(get_VelocityX, set_VelocityX)
    VelocityY = property(get_VelocityY, set_VelocityY)
    VelocityZ = property(get_VelocityZ, set_VelocityZ)
    AlphaV = property(get_AlphaV, set_AlphaV)
    V2 = property(get_V2, set_V2)


    def set_AlphaW(self, x):
        self.AlphaV = x
    def set_W2(self, x):
        self.W2 = x
    def set_Uaxis(self, x):
        self.Uaxis = x
    def set_WelocityX(self, x):
        self.WelocityX = x
    def set_WelocityY(self, x):
        self.WelocityY = x
    def set_WelocityZ(self, x):
        self.WelocityZ = x

    def get_Uaxis(self):
        if self._Uaxis is None:
            rst = self.rotateSpeed * 2 * np.pi /60
            rst = np.tile(self.grid[None, :, :, 1], [self.num, 1, 1]) * rst
            self._Uaxis = rst
            return rst
        else:
            return self._Uaxis
    def get_WelocityX(self):
        if self._WelocityX is None:
            rst = self.VelocityX
            self._WelocityX = rst
            return rst
        else:
            return self._WelocityX
    def get_WelocityY(self):
        if self._WelocityY is None:
            rst = self.VelocityY + self.Uaxis
            self._WelocityY = rst
            return rst
        else:
            return self._WelocityY
    def get_WelocityZ(self):
        if self._WelocityZ is None:
            rst = self.VelocityZ
            self._WelocityZ = rst
            return rst
        else:
            return self._WelocityZ
    def get_AlphaW(self):
        if self._AlphaW is None:
            return np.arctan(self.WelocityY / self.WelocityX)
        else:
            return self._AlphaW
    def get_W2(self):
        if self._W2 is None:
            if "W2" in self.inputDict.keys():
                rst = self.data_2d[..., self.inputDict["W2"]]
            else:
                rst = np.power(self.WelocityX,2) + np.power(self.WelocityY,2) + np.power(self.WelocityZ,2)
            self._W2 = rst
            return rst
        else:
            return self._W2

    Uaxis = property(get_Uaxis, set_Uaxis)
    WelocityX = property(get_WelocityX, set_WelocityX)
    WelocityY = property(get_WelocityY, set_WelocityY)
    WelocityZ = property(get_WelocityZ, set_WelocityZ)
    AlphaW = property(get_AlphaW, set_AlphaW)
    W2 = property(get_W2, set_W2)
    #
    # ===============================================================#
    # 其他物理场参数获取：1.总压；2.总温；3.速度x；4.速度y
    # 关键性能参数获取：1.总压比；2.效率；3.总压损失；4.D因子
    # ===============================================================#
    #
    def set_PressureTotalV(self, x):
        self.PressureTotalV = x
    def set_TemperatureTotalV(self, x):
        self.TemperatureTotalV = x
    def set_PressureRatioV(self, x):
        self.PressureRatioV = x
    def set_TemperatureRatioV(self, x):
        self.TemperatureRatioV = x

    def set_PressureTotalW(self, x):
        self.PressureTotalW = x
    def set_TemperatureTotalW(self, x):
        self.TemperatureTotalW = x

    def set_PressureTotalRot(self, x):
        self.PressureTotalRot = x
    def set_TemperatureTotalRot(self, x):
        self.TemperatureTotalRot = x
    def set_TemperatureIsentropic(self, x):
        self.TemperatureIsentropic = x


    def set_PressureRatioW(self, x):
        self.PressureRatioW = x
    def set_TemperatureRatioW(self, x):
        self.TemperatureRatioW = x

    def set_Efficiency(self, x):
        self.Efficiency = x
    def set_EfficiencyPoly(self, x):
        self.EfficiencyPoly = x
    def set_EfficiencyR(self, x):
        self.EfficiencyR = x
    def set_PressureLoss(self, x):
        self.PressureLoss = x
    def set_DFactor(self, x):
        self.DFactor = x

    def set_PressureLossR(self, x):
        self.PressureLossR = x
    def set_DFactorR(self, x):
        self.DFactorR = x

    def set_EntropyStatic(self, x):
        self.EntropyStatic = x
    def set_EntropyStaticNorm(self, x):
        self.EntropyStaticNorm = x
    def set_MachIsentropic(self, x):
        self.MachIsentropic = x
    def set_Load(self, x):
        self.Load = x
    def set_LoadR(self, x):
        self.LoadR = x



    def get_PressureTotalV(self):
        if self._PressureTotalV is None:
            if "PressureTotalV" in self.inputDict.keys():
                rst = self.data_2d[..., self.inputDict["PressureTotalV"]]
            else:
                rst = self.PressureStatic * \
                      np.power(self.TemperatureTotalV / self.TemperatureStatic, self.kappa / (self.kappa - 1))
            self._PressureTotalV = rst
            return rst
        else:
            return self._PressureTotalV
    def get_TemperatureTotalV(self):
        if self._TemperatureTotalV is None:
            if "TemperatureTotalV" in self.inputDict.keys():
                rst = self.data_2d[..., self.inputDict["TemperatureTotalV"]]
            else:
                rst = self.TemperatureStatic + 0.5 * self.V2 / self.Cp
            self._TemperatureTotalV = rst
            return rst
        else:
            return self._TemperatureTotalV
    def get_PressureRatioV(self):
        if self._PressureRatioV is None:
            rst = self.PressureTotalV / np.tile(self.PressureTotalV[..., :1], [1, 1, self.n_1d])
            self._PressureRatioV = rst
            return rst
        else:
            return self._PressureRatioV
    def get_TemperatureRatioV(self):
        if self._TemperatureRatioV is None:
            rst = self.TemperatureTotalV / np.tile(self.TemperatureTotalV[..., :1], [1, 1, self.n_1d])
            self._TemperatureRatioV = rst
            return rst
        else:
            return self._TemperatureRatioV


    def get_PressureTotalW(self):
        if self._PressureTotalW is None:
            if "PressureTotalW" in self.inputDict.keys():
                rst = self.data_2d[..., self.inputDict["PressureTotalW"]]
            else:
                rst = self.PressureStatic * \
                      np.power(self.TemperatureTotalW / self.TemperatureStatic, self.kappa / (self.kappa - 1))
            self._PressureTotalW = rst
            return rst
        else:
            return self._PressureTotalW
    def get_TemperatureTotalW(self):
        if self._TemperatureTotalW is None:
            if "TemperatureTotalW" in self.inputDict.keys():
                rst = self.data_2d[..., self.inputDict["TemperatureTotalW"]]
            else:
                rst = self.TemperatureStatic + 0.5 * self.W2 / self.Cp
            self._TemperatureTotalW = rst
            return rst
        else:
            return self._TemperatureTotalW

    def get_PressureTotalRot(self):
        if self._PressureTotalRot is None:
            if "PressureTotalRot" in self.inputDict.keys():
                rst = self.data_2d[..., self.inputDict["PressureTotalRot"]]
            else:
                rst = self.PressureStatic * \
                      np.power(self.TemperatureTotalRot / self.TemperatureStatic, self.kappa / (self.kappa - 1))
            self._PressureTotalRot = rst
            return rst
        else:
            return self._PressureTotalRot
    def get_TemperatureTotalRot(self):
        if self._TemperatureTotalRot is None:
            rst = self.TemperatureStatic + 0.5 * self.Uaxis * self.Uaxis / self.Cp
            self._TemperatureTotalRot = rst
            return rst
        else:
            return self._TemperatureTotalRot

    def get_TemperatureIsentropic(self):
        if self._TemperatureIsentropic is None:
            rst = np.tile(self.PressureTotalRot[..., :1], [1, 1, self.n_1d])/self.PressureStatic
            rst = np.power(rst, (self.kappa - 1) / self.kappa)
            rst = 1 / rst * np.tile(self.TemperatureTotalRot[..., :1], [1, 1, self.n_1d])
            self._TemperatureIsentropic = rst
            return rst
        else:
            return self._TemperatureIsentropic

    def get_PressureRatioW(self):
        if self._PressureRatioW is None:
            rst = self.PressureTotalW / np.tile(self.PressureTotalW[..., :1], [1, 1, self.n_1d])
            self._PressureRatioW = rst
            return rst
        else:
            return self._PressureRatioW
    def get_TemperatureRatioW(self):
        if self._TemperatureRatioW is None:
            rst = self.TemperatureTotalW / np.tile(self.TemperatureTotalW[..., :1], [1, 1, self.n_1d])
            self._TemperatureRatioW = rst
            return rst
        else:
            return self._TemperatureRatioW

    def get_Efficiency(self):
        if self._Efficiency is None:
            temp1 = np.abs(np.power(self.PressureRatioV, (self.kappa - 1) / self.kappa) - 1)
            temp2 = np.abs(self.TemperatureRatioV - 1)
            delta = 1e-3
            idx1 = temp1 < delta
            temp1[idx1] = delta
            idx2 = temp2 < delta
            temp2[idx2] = delta

            rst = temp1/temp2

            self._Efficiency = rst
            return rst
        else:
            return self._Efficiency

    def get_EfficiencyPoly(self):
        if self._EfficiencyPoly is None:
            # rst = np.log(self.PressureRatioV) / (np.log(self.TemperatureRatioV) + 1e-5)
            temp1 = np.log(self.PressureTotalV) - np.log(np.tile(self.PressureTotalV[..., :1], [1, 1, self.n_1d]))
            temp2 = np.log(self.TemperatureTotalV) - np.log(np.tile(self.TemperatureTotalV[..., :1], [1, 1, self.n_1d]))
            delta = 1e-4
            idx = temp2 < delta
            temp1[idx] = delta / self.Rg * self.Cp
            temp2[idx] = delta
            rst = temp1/temp2
            rst = rst * self.Rg / self.Cp
            self._EfficiencyPoly = rst
            return rst
        else:
            return self._EfficiencyPoly
    def get_EfficiencyR(self):
        if self._EfficiencyR is None:
            rst = np.abs(np.power(self.PressureRatioW, (self.kappa - 1) / self.kappa) - 1) + 1e-5
            rst = rst / (np.abs(self.TemperatureRatioW - 1) + 1e-5)
            self._EfficiencyR = rst
            return rst
        else:
            return self._Efficiency
    def get_PressureLoss(self):
        if self._PressureLoss is None:
            rst = self.PressureRatioV - 1
            rst = rst / (1 - np.tile(self.PressureStatic[..., :1] / self.PressureTotalV[..., :1], [1, 1, self.n_1d]))
            self._PressureLoss = rst
            return rst
        else:
            return self._PressureLoss

    def get_PressureLossR(self):
        if self._PressureLossR is None:
            rst = 1 - self.PressureRatioW
            rst = rst / (1 - np.tile(self.PressureStatic[..., :1] / self.PressureTotalW[..., :1], [1, 1, self.n_1d]))
            self._PressureLossR = rst
            return rst
        else:
            return self._PressureLossR
    def get_DFactor(self):
        if self._DFactor is None:
            rst = self.VelocityY - np.tile(self.VelocityY[...,  :1], [1, 1, self.n_1d])  # delta V theta 周向速度差
            rst = 0.5 * rst / np.tile(self.MagV[...,  :1], [1, 1, self.n_1d]) / self.sigma
            rst = rst + 1 - self.MagV / np.tile(self.MagV[...,  :1], [1, 1, self.n_1d])
            self._DFactor = rst
            return rst
        else:
            return self._DFactor

    def get_DFactorR(self):
        if self._DFactorR is None:
            rst = self.WelocityY - np.tile(self.WelocityY[...,  :1], [1, 1, self.n_1d])  # delta V theta 周向速度差
            rst = 0.5 * rst / np.tile(self.MagW[...,  :1], [1, 1, self.n_1d]) / self.sigma
            rst = rst + 1 - self.MagW / np.tile(self.MagW[...,  :1], [1, 1, self.n_1d])
            self._DFactorR = rst
            return rst
        else:
            return self._DFactorR
    def get_EntropyStatic(self):
        if self._EntropyStatic is None:
            rst = (1-self.kappa) / self.kappa * (np.log2(self.PressureStatic) - np.log2(101325))
            rst = rst + np.log2(self.TemperatureStatic) - np.log2(288.15)
            rst = self.Cp * rst
            self._EntropyStatic = rst
            return rst
        else:
            return self._EntropyStatic

    def get_EntropyStaticNorm(self):
        if self._EntropyStaticNorm is None:
            rst = -1 * self.EntropyStatic / self.Rg
            rst = np.exp(rst)
            self._EntropyStaticNorm = rst
            return rst
        else:
            return self._EntropyStaticNorm

    def get_MachIsentropic(self):
        if self._MachIsentropic is None:
            rst1 = np.tile(self.PressureTotalRot[..., :1], [1, 1, self.n_1d]) / self.PressureStatic
            rst1 = np.power(rst1, (self.kappa-1) / self.kappa) - 1
            rst2 = self.Uaxis * self.Uaxis / self.kappa / self.Rg / self.TemperatureIsentropic

            rst = np.sqrt(rst1 * 2 / (self.kappa -1 ) + rst2)
            self._MachIsentropic = rst
            return rst
        else:
            return self._MachIsentropic
    def get_Load(self):
        if self._Load is None:
            rst = -self.Cp * (np.tile(self.TemperatureTotalV[..., :1], [1, 1, self.n_1d])
                             - self.TemperatureTotalV) / self.Uaxis / self.Uaxis
            self._Load = rst
            return rst
        else:
            return self._Load

    def get_LoadR(self):
        if self._LoadR is None:
            rst = -self.Cp * (np.tile(self.TemperatureTotalW[..., :1], [1, 1, self.n_1d])
                             - self.TemperatureTotalW) / self.Uaxis / self.Uaxis
            self._LoadR = rst
            return rst
        else:
            return self._LoadR


    PressureTotalV = property(get_PressureTotalV, set_PressureTotalV)
    TemperatureTotalV = property(get_TemperatureTotalV, set_TemperatureTotalV)
    PressureRatioV = property(get_PressureRatioV, set_PressureRatioV)
    TemperatureRatioV = property(get_TemperatureRatioV, set_TemperatureRatioV)

    PressureTotalW = property(get_PressureTotalW, set_PressureTotalW)
    TemperatureTotalW = property(get_TemperatureTotalW, set_TemperatureTotalW)

    PressureTotalRot = property(get_PressureTotalRot, set_PressureTotalRot)
    TemperatureTotalRot = property(get_TemperatureTotalRot, set_TemperatureTotalRot)
    TemperatureIsentropic = property(get_TemperatureIsentropic, set_TemperatureIsentropic)

    PressureRatioW = property(get_PressureRatioW, set_PressureRatioW)
    TemperatureRatioW = property(get_TemperatureRatioW, set_TemperatureRatioW)


    Efficiency = property(get_Efficiency, set_Efficiency)
    EfficiencyPoly = property(get_EfficiencyPoly, set_EfficiencyPoly)
    EfficiencyR = property(get_EfficiencyR, set_EfficiencyR)
    PressureLoss = property(get_PressureLoss, set_PressureLoss)
    PressureLossR = property(get_PressureLossR, set_PressureLossR)
    DFactor = property(get_DFactor, set_DFactor)
    DFactorR = property(get_DFactorR, set_DFactorR)
    EntropyStatic = property(get_EntropyStatic, set_EntropyStatic)
    EntropyStaticNorm = property(get_EntropyStaticNorm, set_EntropyStaticNorm)
    MachIsentropic = property(get_MachIsentropic, set_MachIsentropic)
    Load = property(get_Load, set_Load)
    LoadR = property(get_LoadR, set_LoadR)

    def get_MassFlow(self):
        hub_out = 0.1948
        shroud_out = 0.2370
        MassFlow = self.span_space_average(self.DensityFlow[:, :, -1]) * (
                    shroud_out ** 2 - hub_out ** 2) * np.pi
        return MassFlow[:, np.newaxis]


if __name__ == "__main__":
    grid = np.random.rand(64,64,2)
    post = Post_2d(np.random.rand(64,64,5),grid)
    print(post.PressureStatic[:,0])