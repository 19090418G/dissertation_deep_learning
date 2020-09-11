import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import tqdm, os

original_data = np.array(pd.read_csv('original_data - 00.csv'))
label_data = np.array(pd.read_csv('label.csv'))[:, 24]

for i in tqdm.tqdm(range(400, len(original_data))):
    a = np.array(original_data[i - 400:i, 1], dtype=np.float)
    b = np.array((original_data[i - 400:i, 1] + original_data[i - 400:i, 2]) / 2, dtype=np.float)
    c = np.array(original_data[i - 400:i, 2], dtype=np.float)
    d = original_data[i - 400:i, -1]
    temp = np.array([a, b, c, d])
    np.save('temp.npy', temp)
    os.system('python transform_img.py --label {} --num {}'.format(label_data[i - 400], i - 400))