import os, cv2, tqdm
import numpy as np

from dataset import DataGenerator
from MobileNetV2 import MobileNetv2

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint


img_path = sorted(os.listdir('img'), key=lambda y: int(y.split('_')[0]))

length = int(0.9 * len(img_path))
train_imgpath, test_imgpath = img_path[:length], img_path[length:]

train_generator = DataGenerator(train_imgpath)
valid_generator = DataGenerator(test_imgpath)

model = MobileNetv2((224, 224, 3), 2)

model.summary()
model.compile(optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
model.fit_generator(train_generator, validation_data=valid_generator, epochs=100, verbose=2,
                    callbacks=[
                        ReduceLROnPlateau(factor=0.8, patience=10, verbose=2),
                        EarlyStopping(patience=20),
                        CSVLogger('train.log'),
                        ModelCheckpoint('model.h5', save_best_only=True, monitor='val_acc', verbose=2)
                    ])
