# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:57:43 2025

@author: dell
"""

import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

analysisfolder = r"G:/data"
csvFilesFullPath = glob.glob(os.path.join(analysisfolder, '*.csv'))

print(csvFilesFullPath)

for m in range(len(csvFilesFullPath)):

    print(f'Now folder is csvFilesFullPath{m}')
    rawdata = pd.read_csv(csvFilesFullPath[m], low_memory=False)
    bodyparts = rawdata.iloc[0, 1::3].values
    coordinates = {}
    for i in range(len(bodyparts)):
        tmp = {}
        tmp['x'] = rawdata.iloc[2:, 3*i+1].values.astype(float)
        tmp['y'] = -rawdata.iloc[2:, 3*i+2].values.astype(float)
        tmp['reliability'] = rawdata.iloc[2:, 3*i+3].values.astype(float)
        coordinates[bodyparts[i]] = tmp
    
    reliability_thresh = 0.99
    rlb_idx = {}
    for i in range(len(bodyparts)):
        rlb_idx[bodyparts[i]] = np.where(coordinates['pupil_top']['reliability']>=0.99)[0]

    fps = 90
    time_idx = np.arange(0, len(coordinates['pupil_top']['x']))
    time_span = time_idx / fps
    time_span_r = time_span[rlb_idx['pupil_top']]
    
    pupil_top_x = coordinates['pupil_top']['x']
    pupil_top_x_r = pupil_top_x[rlb_idx['pupil_top']]
    
    pupil_top_y = coordinates['pupil_top']['y']
    pupil_top_y_r = pupil_top_y[rlb_idx['pupil_top']]
    
    pupil_bottom_x = coordinates['pupil_bottom']['x']
    pupil_bottom_x_r = pupil_bottom_x[rlb_idx['pupil_bottom']]
    
    pupil_bottom_y = coordinates['pupil_bottom']['y']
    pupil_bottom_y_r = pupil_bottom_y[rlb_idx['pupil_bottom']]
    # pupil_dist = (((coordinates['pupil_top']['x'])-(coordinates['pupil_bottom']['x']))**2+((coordinates['pupil_top']['y'])-(coordinates['pupil_bottom']['y']))**2)**0.5
    pupil_dist = ((pupil_top_x_r - pupil_bottom_x_r)**2 + (pupil_top_y_r - pupil_bottom_y_r)**2)**0.5
    
    fig, ax = plt.subplots()
    plt.plot(time_span_r, pupil_dist)
    plt.show()