import pandas as pd
import matplotlib.pyplot as plt

one_data = pd.read_csv('train.log')
one_acc = one_data['acc']
one_loss = one_data['loss']
one_val_acc = one_data['val_acc']
one_val_loss = one_data['val_loss']

plt.figure()

plt.subplot(2, 2, 1)
plt.plot(one_loss, label='one')
# plt.legend()
plt.title('loss')

plt.subplot(2, 2, 2)
plt.plot(one_acc, label='one')
# plt.legend()
plt.title('acc')

plt.subplot(2, 2, 3)
plt.plot(one_val_loss, label='one')
# plt.legend()
plt.title('val_loss')

plt.subplot(2, 2, 4)
plt.plot(one_val_acc, label='one')
# plt.legend()
plt.title('val_acc')

plt.show()
