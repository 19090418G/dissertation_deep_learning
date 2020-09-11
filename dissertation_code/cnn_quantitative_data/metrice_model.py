import pandas as pd
import numpy as np

from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

data = pd.read_csv('processing_data.csv')
x, y = np.array(data.iloc[:, 1:71]), np.array(data.iloc[:, 71])

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=42)

x_test = np.expand_dims(x_test, axis=-1)

print('CNN Model One:')
model_one = load_model('model_one.h5')
pred = np.argmax(model_one.predict(x_test), axis=1)
print(classification_report(y_test, pred))
print('AUC:{:.5f}'.format(roc_auc_score(y_test, pred)))
print('ACCURACY:{:.5f}'.format(accuracy_score(y_test, pred)))

print('CNN Model Two:')
model_one = load_model('model_two.h5')
pred = np.argmax(model_one.predict(x_test), axis=1)
print(classification_report(y_test, pred))
print('AUC:{:.5f}'.format(roc_auc_score(y_test, pred)))
print('ACCURACY:{:.5f}'.format(accuracy_score(y_test, pred)))

print('CNN Model Three:')
model_one = load_model('model_three.h5')
pred = np.argmax(model_one.predict(x_test), axis=1)
print(classification_report(y_test, pred))
print('AUC:{:.5f}'.format(roc_auc_score(y_test, pred)))
print('ACCURACY:{:.5f}'.format(accuracy_score(y_test, pred)))