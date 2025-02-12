from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from download import load_data


# build autoencoder for digit prediction
def build_autoencoder(input_shape, dropout_rate=0.2, l2_reg=0.01):

    # conv image to 1-D
    input_img = Input(shape=(input_shape,))

    # Train relu network with 2 layers for encoding
    encoded = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(input_img)
    encoded = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(encoded)
    # regulation using Dropout
    encoded = Dropout(dropout_rate)(encoded)

    # 1 layer relu for decoding
    decoded = Dense(input_shape, activation='relu')(encoded)

    # build autoencoder
    autoencoder = Model(input_img, decoded)

    # build it on mean squared error function
    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

    return autoencoder

# load MNIST train data
x_train, y_train = load_data('mnist_train.pkl')

# for each digit 0-9 creat autoencoder
for digit in range(10):

    # train it on digit [digit]
    digit_indices = y_train == digit
    x_digit = x_train[digit_indices]

    # build the autoencoder
    autoencoder = build_autoencoder(x_train.shape[1])
    autoencoder.fit(x_digit, x_digit, epochs=1000, batch_size=100, shuffle=True)

    # save it
    model_path = f'autoencoder_digit_{digit}.h5'
    autoencoder.save(model_path)



