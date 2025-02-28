import pickle
import matplotlib.pyplot as plt

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# load obj from pickle
def load_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

odimgs, positions, ylabels = load_data('Object_detection.pkl')

predictions = []
for image in odimgs:
    plt.imshow(image, cmap='gray')
    plt.show()

    autoencoder = tf.keras.models.load_model('Object_Detection.h5')

    step_size = 28
    h, w = image.shape

    min_diff = float('inf')
    min_patch = None
    min_x, min_y = 0, 0

    differences = []
    for y in range(0, h - step_size + 1):
        for x in range(0, w - step_size + 1):
            patch = image[y:y + step_size, x:x + step_size]
            patch_flat = patch.flatten().astype(np.float32) / 255.0
            patch_flat = np.expand_dims(patch_flat, axis=0)

            reconstructed = autoencoder.predict(patch_flat)
            reconstructed = reconstructed.reshape((28, 28)) * 255.0

            diff = np.abs(patch - reconstructed)
            mean_diff = diff.mean()
            differences.append(mean_diff)

            if mean_diff < min_diff:
                min_diff = mean_diff
                min_patch = patch.copy()
                min_x, min_y = x, y

    print(min_x, " ", min_y)
    plt.imshow(  image[min_y:min_y+28, min_x:min_x+28], cmap='gray')
    plt.show()
