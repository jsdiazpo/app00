import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

header = st.container()
block1 = st.container()
block2 = st.container()
block3 = st.container()


with header:
    st.title('new app')





with block1:
    st.header('block1')
    st.markdown('bla '*100)
    b1l, b1r = st.columns(2)
    # left column
    mu    = b1l.slider('mu', min_value=-10, max_value=10, value=0, step=1)
    sigma = b1l.slider('sigma', min_value=0.1, max_value=10., value=.5, step=.5)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
    x_axis = np.linspace(-10, 10, 100)
    ax.plot(x_axis, stats.norm.pdf(x_axis, mu, sigma))
    ax.set(ylim=[0,.5])
    ax.grid(zorder=0, alpha=0.25, lw=0.5)


    # right column
    b1r.pyplot(fig)



