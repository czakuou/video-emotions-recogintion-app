from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf

class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, classes, reg=1e-5):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        initializer = EmotionVGGNet.initialize_parameters_he()
        
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        # Block #1
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same",
                         kernel_initializer=initializer,
                         input_shape=inputShape,
                         activation="elu",
                         kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), kernel_initializer=initializer,
                         padding="same", activation="elu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block #2
        model.add(Conv2D(64, (3 ,3), kernel_initializer=initializer,
                         padding="same", activation="elu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), kernel_initializer=initializer,
                         padding="same", activation="elu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block #3
        model.add(Conv2D(128, (3 ,3), kernel_initializer=initializer,
                         padding="same", activation="elu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), kernel_initializer=initializer,
                         padding="same", activation="elu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block #4
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer=initializer,
                        activation="elu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Block #5
        model.add(Dense(64, kernel_initializer=initializer,
                        activation="elu", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Block #6
        model.add(Dense(classes, kernel_initializer=initializer,
                        activation="softmax"))
        
        return model
