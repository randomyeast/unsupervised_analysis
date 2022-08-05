import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np

from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json
import time
import os

# plt.style.use('dark_background')

def plot_log(exp_dir):
    # Load in training and testing losses to plot against eachother
    with open(os.path.join(exp_dir, 'log', 'log.json'), 'r') as f:
        log = json.load(f)
    train_losses = np.array([sum(epoch_log['train']['losses'].values()) for epoch_log in log])
    test_losses = np.array([sum(epoch_log['test']['losses'].values()) for epoch_log in log])

    # Plot a vertical line where the best test loss occurred

    print('something')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.axvline(x=test_losses.argmin(), color='#F97306', label='Best Test Loss')
    plt.legend()
    plt.title('Training / testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(exp_dir, 'log', 'loss.png'))



def connect_mouse_pose(m1x, m1y):

    m1x = np.insert(m1x, 3, m1x[0]).reshape(-1, 1)
    m1x = np.insert(m1x, 4, m1x[2]).reshape(-1, 1)
    m1x = np.insert(m1x, 6, m1x[1]).reshape(-1, 1)
    m1x = np.insert(m1x, 7, m1x[5]).reshape(-1, 1)
    m1x = np.insert(m1x, 10, m1x[5]).reshape(-1, 1)
    m1x = np.insert(m1x, 11, m1x[8]).reshape(-1, 1)
    m1x = np.insert(m1x, 13, m1x[9]).reshape(-1, 1)

    m1y = np.insert(m1y, 3, m1y[0]).reshape(-1, 1)
    m1y = np.insert(m1y, 4, m1y[2]).reshape(-1, 1)
    m1y = np.insert(m1y, 6, m1y[1]).reshape(-1, 1)
    m1y = np.insert(m1y, 7, m1y[5]).reshape(-1, 1)
    m1y = np.insert(m1y, 10, m1y[5]).reshape(-1, 1)
    m1y = np.insert(m1y, 11, m1y[8]).reshape(-1, 1)
    m1y = np.insert(m1y, 13, m1y[9]).reshape(-1, 1)

    return m1x.squeeze(), m1y.squeeze()

def plot_mouse_sequence(seq, path='./gifs/testing/test'):
    """
    Author: Andrew Ulmer
    This expects a N x 14 sequence of MARS keypoints and will output
    a gif of the pose sequence to the path provided
    """

    # Assuming there are seven keypoints
    num_mice = int(seq.shape[-1] / 14)

    # Create figure settings

    fig = plt.figure()
    camera = Camera(fig)
    plt.xlim(-320,320)
    plt.ylim(-240, 240)
    for frame in seq:
        mx, my = frame[::2], frame[1::2]
        plt.scatter(x=mx, y=my, c='b')#  c='#00FFFF')
        mx, my = connect_mouse_pose(mx, my)
        plt.plot(mx, my, c='b') #c='#00FFFF')       
 
        camera.snap()

    ani = camera.animate()
    writer = animation.PillowWriter(fps=20)
    ani.save(f'{path}.gif', writer=writer)

def plot_static_mouse_sequence(seq, path='./gifs/testing/test'):
    """
    Author: Andrew Ulmer
    This expects a N x 14 sequence of MARS keypoints and will output
    a gif of the pose sequence to the path provided
    """

    # Assuming there are seven keypoints
    num_mice = int(seq.shape[-1] / 14)

    # Create figure settings
    plt.xlim(-320,320)
    plt.ylim(-240, 240)
    color = plt.cm.cool(np.linspace(0, 1, len(seq)))
    for frame, c in zip(seq, color):
        mx, my = frame[::2], frame[1::2]
        mx, my = connect_mouse_pose(mx, my)
        plt.plot(mx, my, c=c, alpha=0.25) 

def plot_mouse_reconstruction(original, reconstruction, path='./gifs/testing/test'):
    """
    Author: Andrew Ulmer
    This expects 2 N x 14 sequence of MARS keypoints and will output
    a gif of the pose sequence to the path provided
    """
    # Create figure settings

    fig = plt.figure()
    camera = Camera(fig)
    plt.xlim(-320,320)
    plt.ylim(-240, 240)
    for i, (r, o) in enumerate(zip(reconstruction, original)):
        rx, ry = r[::2], r[1::2]
        ox, oy = o[::2], o[1::2]
   
        if i == 0:
            plt.scatter(x=ox, y=oy, c='#00FFFF', label = 'original')
            plt.scatter(x=rx, y=ry, c='#FF00FF', label = 'reconstruction')

        plt.scatter(x=ox, y=oy, c='#00FFFF')
        plt.scatter(x=rx, y=ry, c='#FF00FF')
        rx, ry = connect_mouse_pose(rx, ry)
        ox, oy = connect_mouse_pose(ox, oy)
        plt.plot(ox, oy, c='#00FFFF')
        plt.plot(rx, ry, c='#FF00FF')       
 
        camera.snap()

    plt.legend()
    ani = camera.animate()
    writer = animation.PillowWriter(fps=20)
    ani.save(f'{path}.gif', writer=writer)

    
