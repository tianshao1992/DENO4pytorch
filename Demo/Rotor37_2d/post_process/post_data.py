import numpy as np
import inspect
import os

class Post_2d(object):
    def __init__(self,data_2d,grid,inputDict=None): #默认输入格式为64*64*5
        self.grid = grid
        self.data_2d = data_2d

        if inputDict is None:
            self.inputDict = {
            "PressureStatic" : 0,
            "TemperatureStatic" : 1,
            "Density" : 2,
            "VelocityX" : 3,
            "VelocityY" : 4
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

        self._PressureRatioV = None
        self._TemperatureRatioV = None

        self._PressureRatioW = None
        self._TemperatureRatioW = None

        self._Efficiency = None
        self._EfficiencyR = None
        self._PressureLoss = None
        self._DFactor = None
        self._PressureLossR = None
        self._DFactorR = None
        self._EntropyStatic = None


    def set_basic_const(self,
                        kappa = 1.403,
                        Cp = 1004,
                        sigma = 1.6,
                        rotateSpeed = -17188 # rpm
                        ):
        self.kappa = kappa
        self.Cp = Cp
        self.sigma = sigma# 代表稠度
        self.rotateSpeed = rotateSpeed




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
    def set_MagV(self, x):
        self.MagV = x


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

    def get_MagV(self):
        if self._MagV is None:
            rst = np.power(self.VelocityX,2) + np.power(self.VelocityY,2) + np.power(self.VelocityZ,2)
            rst = np.sqrt(rst)
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
    MagV = property(get_MagV, set_MagV)


    def set_AlphaW(self, x):
        self.AlphaV = x
    def set_MagW(self, x):
        self.MagV = x
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
    def get_MagW(self):
        if self._MagW is None:
            rst = np.power(self.WelocityX,2) + np.power(self.WelocityY,2) + np.power(self.WelocityZ,2)
            rst = np.sqrt(rst)
            return rst
        else:
            return self._MagW

    Uaxis = property(get_Uaxis, set_Uaxis)
    WelocityX = property(get_WelocityX, set_WelocityX)
    WelocityY = property(get_WelocityY, set_WelocityY)
    WelocityZ = property(get_WelocityZ, set_WelocityZ)
    AlphaW = property(get_AlphaW, set_AlphaW)
    MagW = property(get_MagW, set_MagW)
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
    def set_PressureRatioW(self, x):
        self.PressureRatioW = x
    def set_TemperatureRatioW(self, x):
        self.TemperatureRatioW = x

    def set_Efficiency(self, x):
        self.Efficiency = x
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



    def get_PressureTotalV(self):
        if self._PressureTotalV is None:
            rst = self.PressureStatic * \
                  np.power(self.TemperatureTotalV / self.TemperatureStatic, self.kappa / (self.kappa - 1))
            self._PressureTotalV = rst
            return rst
        else:
            return self._PressureTotalV
    def get_TemperatureTotalV(self):
        if self._TemperatureTotalV is None:
            rst = self.TemperatureStatic + 0.5 * self.MagV * self.MagV / self.Cp
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
                rst = self.TemperatureStatic + 0.5 * self.MagW * self.MagW / self.Cp
            self._TemperatureTotalW = rst
            return rst
        else:
            return self._TemperatureTotalW
    def get_PressureRatioW(self):
        if self._PressureRatioW is None:
            rst = self.PressureTotalW / np.tile(self.PressureTotalW[...,  :1], [1, 1, self.n_1d])
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
            num = int(np.round(self.n_2d / 2))
            rst = np.abs(np.power(self.PressureRatioV[..., num:], (self.kappa - 1) / self.kappa) - 1) + 1e-5
            rst = rst / (np.abs(self.TemperatureRatioV[..., num:] - 1) + 1e-5)
            rst = np.concatenate((np.zeros([self.num, self.n_1d, num]), rst), axis=2)
            self._Efficiency = rst
            return rst
        else:
            return self._Efficiency
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
            rst = rst / (1 - np.tile(self.PressureStatic[...,  :1] / self.PressureTotalV[...,  :1], [1, 1, self.n_1d]))
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
            rst = (1-self.kappa) / self.kappa * np.log2(self.PressureStatic / 101325)
            rst = rst + np.log2(self.TemperatureStatic / 288.15)
            rst = self.Cp * rst
            self._EntropyStatic = rst
            return rst
        else:
            return self._EntropyStatic

    PressureTotalV = property(get_PressureTotalV, set_PressureTotalV)
    TemperatureTotalV = property(get_TemperatureTotalV, set_TemperatureTotalV)
    PressureRatioV = property(get_PressureRatioV, set_PressureRatioV)
    TemperatureRatioV = property(get_TemperatureRatioV, set_TemperatureRatioV)

    PressureTotalW = property(get_PressureTotalW, set_PressureTotalW)
    TemperatureTotalW = property(get_TemperatureTotalW, set_TemperatureTotalW)
    PressureRatioW = property(get_PressureRatioW, set_PressureRatioW)
    TemperatureRatioW = property(get_TemperatureRatioW, set_TemperatureRatioW)

    Efficiency = property(get_Efficiency, set_Efficiency)
    EfficiencyR = property(get_EfficiencyR, set_EfficiencyR)
    PressureLoss = property(get_PressureLoss, set_PressureLoss)
    PressureLossR = property(get_PressureLossR, set_PressureLossR)
    DFactor = property(get_DFactor, set_DFactor)
    DFactorR = property(get_DFactorR, set_DFactorR)
    EntropyStatic = property(get_EntropyStatic, set_EntropyStatic)


if __name__ == "__main__":
    grid = np.random.rand(64,64,2)
    post = Post_2d(np.random.rand(64,64,5),grid)
    print(post.PressureStatic[:,0])