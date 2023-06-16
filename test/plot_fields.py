import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_fields(field,pred,varname):
    plt.rc("font", size=16, family='serif')
    N = field.shape[-1]
    idx = np.random.randint(field.shape[0])
    time_steps = np.arange(10,N,(N-10)//3 - 1)
    print(time_steps)
    fig, axs = plt.subplots(2,4,figsize=(20,10))
    fig.suptitle('{} - Ground truth vs. Prediction'.format(varname),fontsize=24)
    ax = axs[0,0]
    cm = ax.imshow(field[idx,...,time_steps[0]], cmap=sns.cm.icefire)
    ax.set_title('t = ' + str(time_steps[0]) + '$ \Delta t$')
    ax.set_ylabel('Ground truth')
    ax = axs[0,1]
    cm = ax.imshow(field[idx,...,time_steps[1]], cmap=sns.cm.icefire)
    ax.set_title('t = ' + str(time_steps[1]) + '$ \Delta t$')
    ax = axs[0,2]
    cm = ax.imshow(field[idx,...,time_steps[2]], cmap=sns.cm.icefire)
    ax.set_title('t = ' + str(time_steps[2]) + '$ \Delta t$')
    ax = axs[0,3]
    cm = ax.imshow(field[idx,...,time_steps[3]], cmap=sns.cm.icefire)
    ax.set_title('t = ' + str(time_steps[3]) + '$ \Delta t$')
    ax = axs[1,0]
    cm = ax.imshow(pred[idx,...,time_steps[0]], cmap=sns.cm.icefire)
    ax.set_ylabel('Prediction')
    ax = axs[1,1]
    cm = ax.imshow(pred[idx,...,time_steps[1]], cmap=sns.cm.icefire)
    ax = axs[1,2]
    cm = ax.imshow(pred[idx,...,time_steps[2]], cmap=sns.cm.icefire)
    ax = axs[1,3]
    cm = ax.imshow(pred[idx,...,time_steps[3]], cmap=sns.cm.icefire)
    fig.colorbar(cm, ax = axs)
    plt.close()
    return fig
