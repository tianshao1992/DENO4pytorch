# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/8/31 19:00
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：check_data.py
@File ：check_data.py
"""
import os
import sys

# add .py path
file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(file_path.split('Demo')[0]))

import numpy as np
import matplotlib.pyplot as plt
from Utilizes.visual_data import MatplotlibVision

work_path = 'work'
Visual = MatplotlibVision(work_path, input_name=('x', 'y', 'z'), field_name=('u', 'v', 'w'))

all_data = np.load('data/vel_121-140g_600p_gap200_LES64.npy')  #

print([all_data.shape[0], all_data.shape[1]])

total_infer_steps = 500
case_id = 10
s = all_data.shape[2]

print(np.max(all_data, axis=(0, 1, 2, 3, 4)))
print(np.min(all_data, axis=(0, 1, 2, 3, 4)))

for time_id in range(5, total_infer_steps, 10):
    fig, axs = plt.subplots(3, 3, figsize=(25, 25), num=1, layout='constrained')
    Visual.plot_fields_ms(fig, axs, all_data[case_id, time_id, :, :, s // 2, :],
                          all_data[case_id, time_id, :, s // 2, :, :],
                          titles=['z=1/2', 'y=1/2', 'error'])
    title = 'case_{:d}_step_{:d}'.format(case_id, time_id)
    print(title)
    fig.suptitle(title)
    fig.savefig(os.path.join(work_path, 'velocity_case_{}_step_{}.jpg'.format(case_id, time_id)))
    plt.close(fig)