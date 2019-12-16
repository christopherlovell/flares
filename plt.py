
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fancy = lambda x: r'$\rm '+x.replace(' ','\ ')+'$'
ml = lambda x: r'$\rm '+x+'$'

rcParams = {}
rcParams['savefig.dpi'] = 300
rcParams['path.simplify'] = True

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'stixsans'
rcParams['text.usetex'] = False
rcParams['font.size'] = 9
rcParams['mathtext.fontset'] = 'stixsans'

rcParams['axes.linewidth'] = 0.5

rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3
rcParams['xtick.minor.size'] = 1.5
rcParams['ytick.minor.size'] = 1.5
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.minor.visible'] = True
rcParams['xtick.minor.visible'] = True
rcParams['xtick.major.width'] = 0.25
rcParams['ytick.major.width'] = 0.25
rcParams['xtick.minor.width'] = 0.25
rcParams['ytick.minor.width'] = 0.25

rcParams['grid.alpha'] = 0.1
rcParams['grid.color'] = 'k'
rcParams['grid.linestyle'] = '-'
rcParams['grid.linewidth'] = 0.8


# --- legend

# legend.borderaxespad: 0.5
# legend.borderpad: 0.4
# legend.columnspacing: 2.0
# legend.edgecolor: 0.8
# legend.facecolor: inherit
rcParams['legend.fancybox'] = False
rcParams['legend.fontsize'] = 8
# legend.framealpha: 0.8
rcParams['legend.frameon'] = False
# legend.handleheight: 0.7
# legend.handlelength: 2.0
# legend.handletextpad: 0.8
# legend.labelspacing: 0.5
# legend.loc: best
# legend.markerscale: 1.0
# legend.numpoints: 1
# legend.scatterpoints: 1
# legend.shadow: False
# legend.title_fontsize: None


mpl.rcParams.update(rcParams)


def single():

    fig = plt.figure(figsize = (3.5, 3.5))

    left  = 0.15
    height = 0.8
    bottom = 0.15
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    return fig, ax
