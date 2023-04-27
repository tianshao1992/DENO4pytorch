import numpy as np
import os
def get_grid():
    xx = np.linspace(-0.127, 0.126, 64)
    xx = np.tile(xx, [64,1])

    hub_file = os.path.join('data', 'hub_lower.txt')
    hub = np.loadtxt(hub_file)
    shroud_files = os.path.join('data', 'shroud_upper.txt')
    shroud = np.loadtxt(shroud_files)

    yy = []
    for i in range(64):
        yy.append(np.linspace(hub[i],shroud[i],64))

    yy = np.concatenate(yy, axis=0)
    yy = yy.reshape(64, 64).T
    xx = xx.reshape(64, 64)

    return np.concatenate([xx[:,:,np.newaxis],yy[:,:,np.newaxis]],axis=2)
class Post_2d(object):

    def __init__(self,data_2d,grid): #默认输入格式为64*64*5
        self.grid = grid
        self.data_2d = data_2d
        self.set_basic_const()
        self.n_1d = self.data_2d.shape[0]
        self.n_2d = self.data_2d.shape[1]

        # self.PressureStatic= None
        # self.TemperatureStatic = None
        # self.Density = None
        # self.AlphaV = None
        # self.MagV = None
        #
        # self.PressureTotal = None
        # self.TemperatureTotal = None
        # self.Vx = None
        # self.Vy = None
        #
        # self.PressureRatio = None
        # self.TemperatureRatio = None
        # self.Efficiency = None
        # self.PressureLoss = None
        # self.DFactor = None


    def set_basic_const(self,
                        kappa = 1.403,
                        Cp = 1004,
                        sigma = 1.6 #代表稠度
                        ):
        self.kappa = kappa
        self.Cp = Cp
        self.sigma = sigma



    def get_parameter(self,shape_index=None): #计算各种性能参数,shape表示针对哪一部分进行计算
        if shape_index==None:
            # 如果不设置，就对全局进行计算
            shape_index = [np.arange(self.n_1d),np.arange(self.n_2d)]

    def span_density_average(self,data,shape_index): #输入为2维ndarry
        if len(data.shape)<2:
            print("error input in function span_density_average.")

        density_norm = self.Density[shape_index]\
                       /np.tile(np.mean(self.Density[shape_index],axis=0),shape_index.shape[1])
        return data * density_norm

#=================================================================================#
#各类具体参数设置

    #===============================================================#
    #基本参数获取：1.静压；2.静温；3.密度；4.速度方向；5.速度大小
    def get_PressureStatic(self):
        return self.data_2d[:,:,0]
    def get_TemperatureStatic(self):
        return self.data_2d[:,:,1]
    def get_Density(self):
        return self.data_2d[:,:,2]
    def get_AlphaV(self):
        return self.data_2d[:,:,3]
    def get_MagV(self):
        return self.data_2d[:,:,4]

    PressureStatic = property(get_PressureStatic)
    TemperatureStatic = property(get_TemperatureStatic)
    Density = property(get_Density)
    AlphaV = property(get_AlphaV)
    MagV = property(get_MagV)
    #
    # # ===============================================================#
    # # 其他物理场参数获取：1.总压；2.总温；3.速度x；4.速度y
    def get_PressureTotal(self):
        rst = self.PressureStatic * np.power(self.TemperatureTotal/self.TemperatureStatic,self.kappa/(self.kappa-1))
        return rst

    def get_TemperatureTotal(self):
        rst = self.TemperatureStatic + 0.5*self.MagV*self.MagV/self.Cp
        return rst

    def get_Vx(self):
        rst = self.MagV * np.cos(self.AlphaV)
        return rst

    def get_Vy(self):
        rst = self.MagV * np.sin(self.AlphaV)
        return rst

    PressureTotal = property(get_PressureTotal)
    TemperatureTotal = property(get_TemperatureTotal)
    Vx = property(get_Vx)
    Vy = property(get_Vy)
    #
    # # ===============================================================#
    # # 关键性能参数获取：1.总压比；2.效率；3.总压损失；4.D因子
    def get_PressureRatio(self): #流场中任意位置与入口的总压比
        rst = self.PressureTotal / np.tile(self.PressureTotal[:,0],[self.n_1d,1])
        return rst

    def get_TemperatureRatio(self): #流场中任意位置与入口的总压比
        rst = self.TemperatureTotal / np.tile(self.TemperatureTotal[:,0],[self.n_1d,1])
        return rst

    def get_Efficiency(self):
        num = int(np.round(self.n_2d/2))
        rst = np.abs(np.power(self.PressureRatio[:,num:],(self.kappa-1)/self.kappa)-1)+1e-5
        rst = rst/(np.abs(self.TemperatureRatio[:,num:]-1) + 1e-5)
        rst = np.concatenate((np.zeros([self.n_1d,num]),rst),axis=1)
        return rst

    def get_PressureLoss(self):
        rst = self.PressureRatio - 1
        rst = rst / (1-np.tile(self.PressureStatic[:,0]/self.PressureTotal[:,0],[self.n_1d,1]))
        return rst

    def get_DFactor(self):
        rst = self.Vy - np.tile(self.Vy[:,0],[self.n_1d,1]) #delta V theta 周向速度差
        rst = 0.5 * rst / np.tile(self.MagV[:,0],[self.n_1d,1]) /self.sigma
        rst = rst + 1 - self.MagV / np.tile(self.MagV[:,0],[self.n_1d,1])
        return rst


    PressureRatio = property(get_PressureRatio)
    TemperatureRatio = property(get_TemperatureRatio)
    Efficiency = property(get_Efficiency)
    PressureLoss = property(get_PressureLoss)
    DFactor = property(get_DFactor)


if __name__ == "__main__":
    grid = get_grid()
    post = Post_2d(np.random.rand(64,64,5),grid)

    print(post.Efficiency[:,0])