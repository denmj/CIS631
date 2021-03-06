from base_model import BaseModel
from base_model_loader import BaseModelLoader
from tensorflow.keras.models import model_from_json
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.layers import Dense, Conv2D, Dropout,BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras.applications import MobileNetV2


class MobNet_Builder(BaseModel):

    def __init__(self, input_shape, num_classes, metric_list):
        super(MobNet_Builder, self).__init__()
        self.build_model(input_shape, num_classes, metric_list)
        self.hist = []

    def build_model(self, input_shape, num_classes, metric_list):

        self.inception = MobileNetV2(include_top=False, alpha=0.5, weights="imagenet", input_shape=input_shape)
        self.model = keras.models.Sequential(self.inception)

        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(Dropout(0.40))
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.40))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.6))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.RMSprop(lr=0.0001),
                           metrics=metric_list)

    def fit(self, x_train, y_train, batch_size, epochs, plot=False):

        self.hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3)

        if plot:

            print(self.hist.history['accuracy'])
            print(self.hist.history['precision_m'])
            print(self.hist.history['recall_m'])
            print(self.hist.history['f1_m'])

            print(self.hist.history['val_accuracy'])
            print(self.hist.history['val_precision_m'])
            print(self.hist.history['val_recall_m'])
            print(self.hist.history['val_f1_m'])
            if plot:
                # Accuracy plot
                plt.plot(self.hist.history['accuracy'])
                plt.plot(self.hist.history['val_accuracy'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                # Loss plot
                plt.plot(self.hist.history['loss'])
                plt.plot(self.hist.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                # Precision plot
                plt.plot(self.hist.history['precision_m'])
                plt.plot(self.hist.history['val_precision_m'])
                plt.title('model precision')
                plt.ylabel('precision')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                # Recall plot
                plt.plot(self.hist.history['recall_m'])
                plt.plot(self.hist.history['val_recall_m'])
                plt.title('model recall')
                plt.ylabel('recall')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                # F1 plot
                plt.plot(self.hist.history['f1_m'])
                plt.plot(self.hist.history['val_f1_m'])
                plt.title('model F1 score')
                plt.ylabel('F1 score')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()

    def evaluate(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def save(self, checkpoint_path):
        file_path = checkpoint_path + 'model.json'
        model_json = self.model.to_json()
        with open(file_path, 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(checkpoint_path+'model.h5')

    def summary(self):
        self.model.summary()


class ModelLoader(BaseModelLoader):
    KLASS = ['Benign', 'Malignant']

    def __init__(self, json_model, model_weights):
        super(ModelLoader, self).__init__(json_model, model_weights)
        print("MNIST model loaded")

    def predict(self, image):
        self.pred = self.loaded_json_model(image)
        return ModelLoader.KLASS[np.argmax(self.pred)]
