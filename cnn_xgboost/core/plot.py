import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def lineplot(x, y, labels, milestones, title='', file_name=''):
	plt.figure(figsize=(20,10))
	
	for i in range(len(x)):
		plt.plot(x[i], y[i], '-o',label=labels[i])
	
	if milestones:
		for m in milestones:
		    plt.axvline(x=m, color='gray', ls='--')

	plt.title(title)
	plt.xlabel('epoch')
	plt.ylabel('avg loss')
	plt.legend()
	plt.tight_layout()
	plt.savefig(file_name)

if __name__ == '__main__':
	import numpy as np
	y = [
		[
		    92.5000, 96.0167, 96.8833, 97.3167, 97.8167, 98.0333, 98.4000, 98.2833, 98.4500, 98.6000, 
		    98.3833, 98.6333, 98.6667, 98.3167, 98.5333, 98.7500, 98.7000, 98.7333, 98.8000, 98.7167, 
		    98.7333, 99.0167, 98.6833, 98.8167, 98.8500, 98.8500, 99.0833, 98.9667, 98.9667, 99.0500, 
		    99.0333, 99.1167, 99.0833, 99.1667, 99.0000, 99.1167, 99.0667, 99.0833, 98.9667, 98.9500, 
		    99.0000, 99.1500, 99.0000, 99.0667, 98.9833, 98.9833, 99.0167, 98.9833, 98.9833, 98.9500,
		],
		[
		    92.4833, 96.0000, 96.8167, 97.5667, 97.9167, 98.1333, 98.5167, 98.3500, 98.6667, 98.5167, 
		    98.7500, 98.8167, 98.7833, 98.7167, 98.7833, 98.7500, 98.6000, 98.7833, 98.7500, 98.9167, 
		    98.7667, 98.8500, 98.8667, 98.9333, 98.9167, 98.9000, 98.9333, 99.0000, 98.9333, 99.1167, 
		    99.1000, 99.1333, 99.1667, 99.0333, 99.0667, 99.1167, 99.1833, 99.2167, 99.1667, 99.2000, 
		    99.2000, 99.1167, 99.1667, 99.2333, 99.1000, 99.1500, 99.1667, 99.1333, 99.2000, 99.2333,
		]
	]

	x = [list(range(1, len(i)+1)) for i in y]
	labels = ['test'+str(i) for i in range(len(x))]

	# lineplot(x, y, labels, [25, 37, 43], 'title', '../data/loss_plot.png')

	# y = np.array(y)
	test = [
		    92.4833, 96.0000, 96.8167, 97.5667, 97.9167, 98.1333, 98.5167, 98.3500, 98.6667, 98.5167, 
		    98.7500, 98.8167, 98.7833, 98.7167, 98.7833, 98.7500, 98.6000, 98.7833, 98.7500, 98.9167, 
		    98.7667, 98.8500, 98.8667, 98.9333, 98.9167, 98.9000, 98.9333, 99.0000, 98.9333, 99.1167, 
		    99.1000, 99.1333, 99.1667, 99.0333, 99.0667, 99.1167, 99.1833, 99.2167, 99.1667, 99.2000, 
		    99.2000, 99.1167, 99.1667, 99.2333, 99.1000, 99.1500, 99.1667, 99.1333, 99.2000, 99.2333,
		]
	# print(np.argmin(y, axis=1))
	print(np.argmax(test))
	print('{}/{:03d}.ckpt'.format('.', np.argmax(test)))
