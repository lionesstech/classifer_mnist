from tensorflow.keras.datasets import mnist
import pickle


#  dump obj to pickle
def save_data(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# load obj from pickle
def load_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# load MNIST Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalized images
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((len(x_test), -1))

# save train and test data
save_data((x_train, y_train), 'mnist_train.pkl')
save_data((x_test, y_test), 'mnist_test.pkl')