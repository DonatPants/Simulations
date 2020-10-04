# calculating the time evolution of N hard wall discrete case using Green's function

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from scipy.special import jv


# parameters of the simulation
N = 101
g = 0.001
t = 0.05 # time scale of one iteration
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

def g_tilde(n, m, t):
    """
    Computes Green's function for time evolution
    :param n: index
    :param m: index
    :param t: time
    :return: g tilde function
    """
    if n <= m:
        return (1j**(m-n))*np.exp(-(n-m-1)*g)*(jv(-(n-m-1), 2*t) - ((-1)**n) * jv(n + m + 1, 2*t))
    elif n > m:
        return (1j**(n-m))*np.exp(-(n-m-1)*g)*(jv((n-m-1), 2*t) + ((-1)**m) * jv(n + m + 1, 2*t))

    # should never happen
    raise TypeError("n and m should be integers, values given: n = " + str(n) + ", m = " + str(m))


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
    x = range(N)

    # compute the discrete wave function y for general initial conditions
    y = []
    for n in range(N):
        sum = 0
        for m in range(N):
            sum += 1j*g_tilde(n,m,i*t)*vec1[m]
        y.append(sum)

    '''
    # if vec1 is 0 everywhere other than for index 0 where it is 1, we can use this computation instead of the loop
    y = [1j*g_tilde(n,0,i*t) for n in range(N)]
    '''

    y = np.array(y)

    y1 = y.real
    y2 = y.imag
    y3 = abs(y)**2

    line[0].set_data(x, y1)
    line[1].set_data(x, y2)
    line[2].set_data(x, y3)

    return line

# animate the plots
anim = animation.FuncAnimation(fig, func=animate, repeat=False, blit=True, frames=600, interval=1)


#save the simulation
'''
path = r"E:\Files\Desktop\test\discrete sims\hard wall case green1.mp4"
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Omer Levy'), bitrate=1800)
anim.save(path, writer=writer)
'''

plt.show()

