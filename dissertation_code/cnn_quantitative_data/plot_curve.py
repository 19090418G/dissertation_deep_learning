import pandas as pd
import matplotlib.pyplot as plt

one_data = pd.read_csv('model_one.log')
one_acc = one_data['acc']
one_loss = one_data['loss']
one_val_acc = one_data['val_acc']
one_val_loss = one_data['val_loss']

two_data = pd.read_csv('model_two.log')
two_acc = two_data['acc']
two_loss = two_data['loss']
two_val_acc = two_data['val_acc']
two_val_loss = two_data['val_loss']

three_data = pd.read_csv('model_three.log')
three_acc = three_data['acc']
three_loss = three_data['loss']
three_val_acc = three_data['val_acc']
three_val_loss = three_data['val_loss']

plt.figure()

plt.subplot(2, 2, 1)
plt.plot(one_loss, label='one')
plt.plot(two_loss, label='two')
plt.plot(three_loss, label='three')
plt.legend()
plt.title('loss')

plt.subplot(2, 2, 2)
plt.plot(one_acc, label='one')
plt.plot(two_acc, label='two')
plt.plot(three_acc, label='three')
plt.legend()
plt.title('acc')

plt.subplot(2, 2, 3)
plt.plot(one_val_loss, label='one')
plt.plot(two_val_loss, label='two')
plt.plot(three_val_loss, label='three')
plt.legend()
plt.title('val_loss')

plt.subplot(2, 2, 4)
plt.plot(one_val_acc, label='one')
plt.plot(two_val_acc, label='two')
plt.plot(three_val_acc, label='three')
plt.legend()
plt.title('val_acc')

plt.show()
