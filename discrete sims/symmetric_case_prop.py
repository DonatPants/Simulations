# calculating the time evolution of N periodic discrete case using the propagator

import matplotlib.pyplot as plt
from scipy.linalg import expm
import numpy as np
import matplotlib.animation as animation


# parameters of the simulation
N = 51
g = 0.001
t = 0.1 # time scale of one iteration
y_range = 0.6 # scale of y axis


def one_hot(N, i):
    """
    Generates a list of length N that is zero everywhere, but 1 at index i
    :param N: length of list
    :param i: index of only non-zero element
    :return: list of size N that is zero everywhere but 1 at index 1
    """
    vec = N*[0]
    vec[i] = 1
    return vec

# generate the matrices for the hamiltonian
Sr = [one_hot(N,N-1)]
for i in range(0,N-1):
    Sr.append(one_hot(N,i))
Sr = np.matrix(Sr)
Sl = Sr.T
H = -(np.exp(-g)*Sr + np.exp(g)*Sl)


# initial vector, zero everywhere, 1 at one point on the lattice
vec1 = np.matrix(one_hot(N,0)).transpose()

# initialize the plotting objects
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
line1, = ax1.plot([], [], color='b')
line2, = ax2.plot([], [], color='r')
line3, = ax3.plot([], [], color='g')
line = [line1, line2, line3]

# set the x and y scale
for ax in [ax1, ax2]:
    ax.set_xlim(-1,N)
    ax.set_ylim(-y_range,y_range)

ax3.set_xlim(-1,N)
ax3.set_ylim(-y_range**2,y_range**2)


def animate(i):
    """
    animate the next frame
    :param i: the number of the frame
    :return: list of functions to animate the plots
    """
    x = range(N)
    # compute propagator
    mat1 = expm(t*i*1j*H)
    # compute discrete wave function at time i*t
    y = np.array(mat1*vec1).T

    # separate into real, imaginary and abs square
    y1 = y.real[0]
    y2 = y.imag[0]
    y3 = abs(y)**2

    line[0].set_data(x, y1)
    line[1].set_data(x, y2)
    line[2].set_data(x, y3)

    return line

# animate the plots
anim = animation.FuncAnimation(fig, func=animate, repeat=False, blit=True, frames=50, interval=1)


#save the simulation
path = r"E:\Files\Desktop\quantum computation\simulations\test1.mp4"
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Omer Levy'), bitrate=1800)
anim.save(path, writer=writer)


# show the simulation
plt.show()




