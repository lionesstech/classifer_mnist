from tensorflow.keras.datasets import mnist
import pickle
import matplotlib.pyplot as plt
import random,cv2

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

# create 4 backgrounds
# measure there sizes
shapes = []
bgs = []
for i in range(1, 5):
    im = cv2.imread(f"bg{i}.jpg", cv2.IMREAD_GRAYSCALE)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    #plt.show()
    bgs.append(im)
    print(im.shape)
    shapes.append(im.shape)

# create 1000 sub images that include 1 digit in random background
odimgs = [] # object detection
positions = [] #positions of digit
ylabels=[]
for i in range(1000):
    random_bg = random.randint(0,3)
    bg = bgs[random_bg].copy()
    shape = shapes[random_bg]
    random_x = random.randint(0, shape[0]-28-1)
    random_y = random.randint(0,shape[1]-28-1)
    random_index = random.randint(0, len(x_train)-1)
    bg[random_x:(random_x+28),random_y:(random_y+28)] = x_train[random_index]
    label = y_train[random_index]
    plt.imshow(bg, cmap='gray')
    plt.show()
    odimgs.append(bg)
    ylabels.append(label)
    positions.append([random_x, random_y])
    # print(random_x," ",random_y , " digit ",label)
save_data((odimgs,positions,ylabels),'Object_detection.pkl')


# normalized images
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((len(x_test), -1))

# save train and test data
save_data((x_train, y_train), 'mnist_train.pkl')
save_data((x_test, y_test), 'mnist_test.pkl')




