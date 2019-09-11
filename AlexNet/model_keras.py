from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, BatchNormalization
from keras.layers import Dropout

class AlexNet():
    def __init__(self):
        model = Sequential()
        model.add(Conv2D(96, (11, 11), input_shape=[227, 227, 3], strides=(4, 4), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))

        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))

        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        model.add(Flatten())

        model.add(Dense(units=4096, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(units=4096, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(units=1000))
        model.summary()

        self.model = model
        #print(self.model)

    def train(self, x_train, y_train):
        model = self.model

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        earlystopper = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint('model-1.h5', verbose=1, save_best_only=True)
        model.fit(x_train, y_train, alidation_split=0.1, batch_size=100, nb_epoch=40,
                  callbacks=[earlystopper, checkpointer])

    def evaluate(self, x_test, y_test):
        model = self.model
        score = model.evaluate(x_test, y_test)
        print("Total loss on Testing data: ", score[0])
        print("Accuracy of Testing data: ", score[1])


def get_model():
    return AlexNet()

if __name__ == '__main__':
    net = AlexNet()
