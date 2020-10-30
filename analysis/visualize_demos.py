import os
import cv2
import sys
import pickle
import numpy as np
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
import time

def show_video(images):
    for image in images:
        cv2.imshow('frame', cv2.resize(image, (500,500)))
        cv2.waitKey(50)
    cv2.destroyAllWindows()


def parse_images(data):
    return [f for ep in data for a in ep['interm_obs'] for f in a]


def visualize_last_demo(path):
    with open(path, 'rb') as fh:
        data = pickle.load(fh)

    images = parse_images(data)
    show_video(images)

if __name__ == "__main__":
    visualize_last_demo(sys.argv[1])


