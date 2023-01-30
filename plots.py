from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import os


def smooth(scalars , weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed



plots_dict = {}
f_idx = 0
for file_tb in os.listdir('/tensorboard'):
    for e in summary_iterator("/tensorboard/" + file_tb):
        x = e.step
        for v in e.summary.value:
            if plots_dict.get(v.tag) is not None:
                plots_dict[v.tag]['y'].append(v.simple_value)
                plots_dict[v.tag]['x'].append(x)
            else:
                plots_dict[v.tag] = {'x': [], 'y': []}

    for k, v in plots_dict.items():
        print(k, len(v['x']))
        print(k, len(v['y']))
        plt.figure(f_idx)
        f_idx += 1
        plt.plot(v['x'], v['y'])
        plt.plot(v['x'], smooth(v['y'], 0.9))
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(k.split('/')[-1])
        plt.savefig('/tensorboard/%s_%s.png' % (file_tb, k.replace('/', '_')))
        plt.close()