import seaborn as sns
from matplotlib.patches import Circle
import numpy as np


def plot_airy(fstop, sensor_width, pixels_in_row, n):
	cols=['r','g','b']
	fig=sns.plt.figure(figsize=(12,12))
	ax=fig.add_subplot(1,1,1)
	for nn,i in enumerate([.000000600,.000000550, .000000450]):
		airy=1.22*fstop*i*2
		p_size=sensor_width/pixels_in_row
		radius=airy/p_size
		circ=sns.plt.Circle((2.5,2.5), radius=radius/2, color=cols[nn], alpha=0.5)
		ax.add_patch(circ)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		sns.plt.title('F-number=%s'%(fstop), fontsize=60)
		sns.plt.xlim(0,5);sns.plt.ylim(0,5)
		sns.plt.xticks(range(5));sns.plt.yticks(range(5))
	sns.plt.savefig('airy%s.png'%(n), dpi=800)

sensor_width=0.0358
pixels_in_row=4368
for n,i in enumerate([2.8, 8 , 16]):
	plot_airy(i, sensor_width, pixels_in_row, n)
