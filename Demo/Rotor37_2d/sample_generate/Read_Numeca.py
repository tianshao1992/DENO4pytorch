def readmf(mffile,keyword):
    f = open(mffile.replace("\\","/"), 'r')
    data = f.readlines()
    f.close()
    # keyList = ['Static_pressure' , 'Mass_flow' , 'Absolute_total_pressure_ratio', 'Isentropic_efficiency']
    for line in data:
        if keyword in line:
            linelist = line.split(' ')
            linelist = [x.strip() for x in linelist if x.strip() != '']

            if len(linelist)<=3:
                return float(linelist[1])

            elif len(linelist)==4:
                return [float(linelist[1]),float(linelist[2])]

if __name__ == "__main__":
    mffile = r"G:\WQN\anto_off_line\rotor37_case\stage_off_line_3\stage_computation_140000\stage_computation_140000.mf"
    mfdata = readmf(mffile, 'Mass_flow')
    Rst = abs((mfdata[0] - mfdata[1]) / mfdata[0])
    print(Rst*100)

