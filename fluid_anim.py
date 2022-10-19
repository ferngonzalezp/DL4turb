import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter 
import matplotlib.pyplot as plt
import seaborn

def fluid_anim(field,name):
  ims = []
  fig, (ax, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]})
  fig.set_size_inches(8,5)
  for i in range(field.shape[-1]):
      im = ax.imshow(field[:,:,i], cmap=seaborn.cm.icefire)
      im2 = ax2.text(0.5, 0.5, "t = {0:.2f} $\delta$t".format(i), size=16, ha="center", color="b")
      ims.append([im, im2])

  ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False,
                                  repeat_delay=1000)

  writer = PillowWriter(fps=8)
  plt.ioff()
  ani.save(name+".gif", writer=writer)