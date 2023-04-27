class Test:
    def __init__(self):
        # self.PressureStatic= None
        # self.TemperatureStatic = None
        # self.Density = None
        # self.VelocityX = None
        # self.VelocityY = None
        # self.VelocityZ = None
        # self.AlphaV = None
        # self.MagV = None

        # self.PressureTotal = None
        # self.TemperatureTotal = None
        # self.PressureRatio = None
        # self.TemperatureRatio = None
        # self.Efficiency = None
        # self.PressureLoss = None
        # self.DFactor = None

        self.Uaxis = None

        self.WelocityX = None
        self.WelocityY = None
        self.WelocityZ = None


if __name__ == '__main__':
    test = Test()
    print(test.__dict__)
    for k in test.__dict__:
        print("def set_" + k + "(self,x):")
        print("\tself." + k + "=x")
    for k in test.__dict__:
        print("def get_" + k + "(self):")
        print("\tif self._" + k + " is None:")
        print("\t\treturn XXX")
        print("\telse:")
        print("\t\treturn self._" + k)
    for k in test.__dict__:
        print(k + "= property(get_" + k + ", set_" + k +")")
