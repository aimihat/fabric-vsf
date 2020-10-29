"""Inspect data that was generated from demo policy, to filter for DDPG and
report demonstrator data for paper.
"""


import os
import cv2
import sys
import pickle
import numpy as np
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
import time



def show_episodes(data):
    """displays episode frames for a given demo pickle"""
    fig, ax = plt.subplots()
    img = plt.imshow(data[0]['interm_obs'][0][0])
    annotation = plt.text(0, 3, 'episode...', fontsize=15, color='white') 
    axbackground = fig.canvas.copy_from_bbox(ax.bbox)
    fig.canvas.draw()
    plt.show(block=False)

    for i_ep, ep in enumerate(data):
        frames = ep['interm_obs']
        for action in frames:
            for i, f in enumerate(action):
                annotation.set_text(f'episode {i_ep} frame {i}')
                img.set_data(f)
                
                fig.canvas.restore_region(axbackground)
                
                ax.draw_artist(img)
                ax.draw_artist(annotation)
                
                fig.canvas.blit(ax.bbox)
                fig.canvas.flush_events()
    
    plt.pause(10)



if __name__ == "__main__":
    path = '/Users/aimihat/Desktop/Thesis/fabric-vsf/logs/demos-2020-10-26-23-56-pol-video_random-seed-22-tier0-model-NA-cost-NA_epis_1.pkl'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)

    show_episodes(data)


