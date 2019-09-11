from distributed.protocol import keras
from keras import Input, Model
from keras.initializers import he_normal
from keras.layers import Conv2D, Activation, BatchNormalization, SeparableConv2D, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Dense
from keras import layers


def Xception(num_classes):
    input = Input(shape=[229, 229, 3])

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x1 = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x1 = BatchNormalization()(x1)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, x1])

    x1 = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, x1])

    x1 = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, x1])

    for i in range(8):
        x1 = x
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = layers.add([x, x1])

    x1 = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x1 = BatchNormalization()(x1)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, x1])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()


if __name__ == '__main__':
    Xception(10000)







