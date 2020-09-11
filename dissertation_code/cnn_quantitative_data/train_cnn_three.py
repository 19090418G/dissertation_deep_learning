import pandas as pd
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import Adam, RMSprop, SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

data = pd.read_csv('processing_data.csv')
x, y = np.array(data.iloc[:, 1:71]), to_categorical(np.array(data.iloc[:, 71]))

# StandardScaler
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# Normalizer
# scaler = Normalizer()
# x = scaler.fit_transform(x)

# MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False, random_state=42)


x_train, x_val, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_val, axis=-1), np.expand_dims(x_test, axis=-1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='tanh', input_shape=(70, 1), kernel_regularizer=l2(0.002)))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=3, activation='tanh', strides=2, kernel_regularizer=l2(0.002)))
model.add(BatchNormalization())

model.add(Conv1D(filters=128, kernel_size=3, activation='tanh', kernel_regularizer=l2(0.002)))
model.add(BatchNormalization())
model.add(Conv1D(filters=128, kernel_size=3, activation='tanh', strides=2, kernel_regularizer=l2(0.002)))
model.add(BatchNormalization())

model.add(Conv1D(filters=256, kernel_size=3, activation='tanh', kernel_regularizer=l2(0.002)))
model.add(BatchNormalization())
model.add(Conv1D(filters=256, kernel_size=3, activation='tanh', strides=2, kernel_regularizer=l2(0.002)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh', kernel_regularizer=l2(0.002)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.002)))

model.summary()

model.compile(optimizer=Adam(0.00005), loss=categorical_crossentropy, metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=2, epochs=100, shuffle=False, batch_size=256, callbacks=[
    CSVLogger('model_three.log'),
    ReduceLROnPlateau(factor=0.8, patience=20, verbose=2),
    ModelCheckpoint('model_three.h5', save_best_only=True, monitor='val_acc', verbose=2)
])

model = load_model('model_three.h5')

pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test, pred))
print('AUC:{:.5f}'.format(roc_auc_score(y_test, pred)))
print('ACCURACY:{:.5f}'.format(accuracy_score(y_test, pred)))