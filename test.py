import tensorflow as tf
import numpy as np
from download import load_data

# load the data from pkl file
x_test, y_test = load_data('mnist_test.pkl')


# calculate the difference
def reconstruction_error(original, reconstructed):
    return np.mean(np.square(original - reconstructed), axis=1)


# predict digit for test set
def predict_digitsfunc(x_test):
    # load the train autoencoders
    autoencoders = [tf.keras.models.load_model(f'autoencoder_digit_{i}.h5') for i in range(10)]

    # calculate the reconstructed images to digit 0-9
    reconstructed_images = [autoencoder.predict(x_test) for autoencoder in autoencoders]

    #  calculate the difference from digit 0-9
    errors = np.array([reconstruction_error(x_test, reconstructed) for reconstructed in reconstructed_images])

    # set digit as min difference per autoencoder
    predicted_digits = np.argmin(errors, axis=0)

    # return predictions
    return predicted_digits


# Predict digits
predicted_digits = predict_digitsfunc(x_test)
# accuracy in %
accuracy = np.mean(predicted_digits == y_test) * 100
print(f"accuracy on test set is: {accuracy:.2f}%")
