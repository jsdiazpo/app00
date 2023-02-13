import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
# for animation
from matplotlib import animation
import streamlit.components.v1 as components

header = st.container()
block1 = st.container()
block2 = st.container()
block3 = st.container()


with header:
    st.title('Testing some plots')




###########################################################################
def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

def func(t, line):
    t = np.arange(0,t,0.1)
    y = np.sin(t)
    line.set_data(t, y)
    return line

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

with block1:
    st.header(f'new block')
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=200, interval=20, blit=True)
    components.html(anim.to_jshtml(), height=1000)

#     # W = 1
#     bl, br = st.columns(2)
#     W_values = ['1 kg', '10 kg', '100 kg']
#     W_value = bl.radio('Select a TNT charge', W_values)
#     W = float(W_value.split()[0])
#     # bl.write('some text', W)


#     df = load_bw_data(W)
#     # st.dataframe(df[['t', 'R']].loc[700:750].round(2))
#     # st.dataframe(df.head().round(2))

#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2))
#     # x_axis = np.linspace(-10, 10, 100)
#     t_scale = 1e3  # milisecs
#     ax.plot(df['t']* t_scale, df['R'], c='tomato', lw=2)
#     # ax.plot(df['tau'], df['z'])
#     ax.set(xlabel='t (ms)', ylabel='R (m)', 
#             xlim=[0,1e-3 * t_scale], ylim=[0,3.5], )
#     # boxs = dict(boxstyle="round", ec='gray', fc='w')
#     ax.grid(zorder=0, alpha=0.25, lw=0.5)
#     br.pyplot(fig)

###########################################################################
P0   = 0.1e6  # atmospheric pressure
rho0 = 1.23   # air density
a0 = np.sqrt(1.4 * P0 / rho0)  # speed of sound
@st.experimental_memo
def load_bw_data(W):
    # W: explosive mass in kg of TNT

    E0 = W * 4.294e6  # conversion from kg TNT to joules
    df = pd.read_csv('data/BlastWaveSolution.csv')
    R0 = (E0/P0)**(1/3) 
    df['R'] = R0 * df['z']
    df['t'] = R0/a0 * df['tau']
    # st.text(f'R0={R0}')
    return df

with block2:

    # W = 1
    st.header(f'Blast Wave for a TNT charge')
    bl, br = st.columns(2)
    W_values = ['1 kg', '10 kg', '100 kg']
    W_value = bl.radio('Select a TNT charge', W_values)
    W = float(W_value.split()[0])
    # bl.write('some text', W)


    df = load_bw_data(W)
    # st.dataframe(df[['t', 'R']].loc[700:750].round(2))
    # st.dataframe(df.head().round(2))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2))
    # x_axis = np.linspace(-10, 10, 100)
    t_scale = 1e3  # milisecs
    ax.plot(df['t']* t_scale, df['R'], c='tomato', lw=2)
    # ax.plot(df['tau'], df['z'])
    ax.set(xlabel='t (ms)', ylabel='R (m)', 
            xlim=[0,1e-3 * t_scale], ylim=[0,3.5], )
    # boxs = dict(boxstyle="round", ec='gray', fc='w')
    ax.grid(zorder=0, alpha=0.25, lw=0.5)
    br.pyplot(fig)



###########################################################################
with block3:
    st.header('Gaussian distribution')
    # st.header('block1')
    # st.markdown('bla '*100)
    bl, br = st.columns(2)
    # left column
    mu    = bl.slider('mu', min_value=-10, max_value=10, value=0, step=1)
    sigma = bl.slider('sigma', min_value=1.0, max_value=5.0, value=1.0, step=.25)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
    x_axis = np.linspace(-10, 10, 100)
    ax.plot(x_axis, stats.norm.pdf(x_axis, mu, sigma))
    ax.set(ylim=[0,.5])
    # boxs = dict(boxstyle="round", ec='gray', fc='w')
    ax.text(3, .43, f'$\mu$={mu}')
    ax.text(3., .38, f'$\sigma$={sigma:.2f}')
    ax.grid(zorder=0, alpha=0.25, lw=0.5)


    # right column
    br.pyplot(fig)

###########################################################################
