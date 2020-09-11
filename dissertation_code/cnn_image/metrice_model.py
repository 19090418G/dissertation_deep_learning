import os
import pandas as pd
import numpy as np

from dataset import DataGenerator
from keras.models import load_model

from keras.applications.mobilenet import relu6, DepthwiseConv2D
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve

model = load_model('model.h5', custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})

img_path = sorted(os.listdir('img'), key=lambda y: int(y.split('_')[0]))
length = int(0.9 * len(img_path))
test_imgpath = img_path[length:]
valid_generator = DataGenerator(test_imgpath)

y, pred = [], []
for idx, (i, j) in enumerate(valid_generator):
    pred_temp = model.predict(i)
    y.extend(list(np.argmax(j, axis=1)))
    pred.extend(list(pred_temp))

    if idx == len(valid_generator):
        break

y, pred = np.array(y), np.array(pred)

pred_temp = np.argmax(pred, axis=1)
roc = roc_auc_score(y, pred_temp)
print(classification_report(y, pred_temp))
print('AUC:{:.5f}'.format(roc))
print('ACCURACY:{:.5f}'.format(accuracy_score(y, pred_temp)))

import matplotlib.pylab as plt
fpr, tpr, thresholds = roc_curve(y, np.max(pred, axis=1))

plt.plot(fpr, tpr, label='auc:{:.5f}'.format(roc))
plt.legend()
plt.show()