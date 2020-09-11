import argparse
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str)
parser.add_argument('--num', type=str)

args = parser.parse_args()

temp = np.load('temp.npy', allow_pickle=True)
a, b, c, d = temp
a, b, c = np.array(a, dtype=np.float), np.array(b, dtype=np.float), np.array(c, dtype=np.float)

plt.subplot(2, 1, 1)
plt.fill_between(np.arange(400), a, b, color='r')
plt.fill_between(np.arange(400), c, b, color='b')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.bar(np.arange(400), d, color='g')
plt.axis('off')

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.axis('off')

plt.savefig('img\{}_{}.jpg'.format(args.num, args.label))
