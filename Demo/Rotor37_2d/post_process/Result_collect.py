# find the figure or text
# change the file name and form
# copy it into a new dir


import os
import shutil
# import cairosvg

if __name__ == "__main__":
    pathSave = r"D:\WQN\CODE\DENO4pytorch-main\Demo\Rotor37_2d\collect/"
    # name = 'MLP'
    filenameList =[
     'work_2700_MSELoss',
     'work_FNO_mode_L1smoothLoss',
     'work_FNO_mode_MSELoss',
     'work_L1smoothLoss',
     'work_MSELoss',
    ]

    fileList = [
                # 'log_loss.svg',
                'valid_box.jpg',
                # 'train_solution_0.jpg', 'train_solution_1.jpg', 'train_solution_2.jpg',
                # 'train_solution_eff_0.jpg', 'train_solution_eff_1.jpg', 'train_solution_eff_2.jpg',
                # 'valid_solution_0.jpg', 'valid_solution_1.jpg', 'valid_solution_2.jpg',
                # 'valid_solution_eff_0.jpg', 'valid_solution_eff_1.jpg', 'valid_solution_eff_2.jpg',
                ]

    for filename in filenameList:
        work_load_path = os.path.join(r"D:\WQN\CODE\DENO4pytorch-main\Demo\Rotor37_2d/",filename)
        workList = os.listdir(work_load_path)
        for name in workList:#['MLP','deepONet','FNO','UNet','Transformer']:
            nameReal = name.split("_")[0]
            modes = 10
            if len(name.split("_"))==2:
                modes = int(name.split("_")[1])
            work_path = os.path.join(work_load_path,name)

            for file in fileList:

                    if os.path.exists(os.path.join(work_path,file)):
                        # if 'svg' in file:
                            # fileNew = file.replace("svg", "png")  # png文件名
                            # cairosvg.svg2png(url=os.path.join(work_path,file), write_to=os.path.join(work_path,fileNew), dpi=1000)
                            # file = fileNew
                        shutil.copy(os.path.join(work_path,file),os.path.join(pathSave,filename+"_"+name+"_"+file))






