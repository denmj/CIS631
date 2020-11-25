import warnings
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image

from tensorflow import keras
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from clf_basic_CNN import CNN_Builder
from resnet_50 import ResNet_Builder
from inception_v3 import Inception_Builder
from mobil_net import MobNet_Builder
from metrics import recall_m, precision_m, f1_m
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


TRAIN_BENIGN_PATH = 'C://Users//denis//Desktop//ML//631_Project//data//train//benign'
TRAIN_MALIGN_PATH = 'C://Users//denis//Desktop//ML//631_Project//data//train//malignant'

TEST_BENIGN_PATH = 'C://Users//denis//Desktop//ML//631_Project//data//test//benign'
TEST_MALIGN_PATH = 'C://Users//denis//Desktop//ML//631_Project//data//test//malignant'
paths = [TRAIN_BENIGN_PATH, TRAIN_MALIGN_PATH, TEST_BENIGN_PATH, TEST_MALIGN_PATH]


def load_dataset(path_list):
    image_read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    data_sets = []
    for path in path_list:
        ims_benign = [image_read(os.path.join(path, filename)) for filename in os.listdir(path)]
        data_sets.append(np.array(ims_benign, dtype='uint8'))
    return data_sets


def show_images(X_, Y_, rows, cols):
    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(top=0.85)

    for i in range(1, rows*cols+1):
        ax = fig.add_subplot(rows, cols, i)
        if Y_[i] == 0:
            ax.title.set_text('not cancer skin')

        else:
            ax.title.set_text('cancer skin')
        plt.imshow(X_[i])
    plt.show()


def preprocess_dataset(datasets):
    pass


s = load_dataset(paths)

image_read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
# Load in training pictures
ims_benign = [image_read(os.path.join(TRAIN_BENIGN_PATH, filename)) for filename in os.listdir(TRAIN_BENIGN_PATH)]
X_benign_train = np.array(ims_benign, dtype='uint8')

ims_malignant = [image_read(os.path.join(TRAIN_MALIGN_PATH, filename)) for filename in os.listdir(TRAIN_MALIGN_PATH)]
X_malignant_train = np.array(ims_malignant, dtype='uint8')

# Load in testing pictures
ims_benign = [image_read(os.path.join(TEST_BENIGN_PATH, filename)) for filename in os.listdir(TEST_BENIGN_PATH)]
X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [image_read(os.path.join(TEST_MALIGN_PATH, filename)) for filename in os.listdir(TEST_MALIGN_PATH)]
X_malignant_test = np.array(ims_malignant, dtype='uint8')


# labels
benign_train_label = np.zeros(len(X_benign_train))
malign_train_label = np.ones(len(X_malignant_train))
benign_test_label = np.zeros(len(X_benign_test))
malign_test_label = np.ones(len(X_malignant_test))



# Merge data
X_train = np.concatenate((X_benign_train, X_malignant_train), axis=0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis=0)
X_test = np.concatenate((X_benign_test, X_malignant_test), axis=0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis=0)



# Shuffle  data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

# show_images(X_train, Y_train, 2, 2)

# Normalize
x_train = X_train.astype("float32") / 255
x_test = X_test.astype("float32") / 255

# convert labels to categorical for Keras
y_train = to_categorical(Y_train, num_classes=2)
y_test = to_categorical(Y_test, num_classes=2)


cnn_model_path = os.getcwd() + '\models\\cnn_model.h5'
resnet_model_path = os.getcwd() + '\models\\resnet50_model.h5'
inception_v3_path = os.getcwd() + '\models\\inceptionv3_model.h5'
mobil_net_path = os.getcwd() + '\models\\mobilnet_model.h5'

num_classes = 2
batch_size = 64
epochs = 250
input_shape = (224, 224, 3)
metrics = ['accuracy', recall_m, precision_m, f1_m]

# CNN training
# model = CNN_Builder(input_shape, num_classes, metrics)
# model.summary()
# model.fit(x_train, y_train, batch_size, epochs, plot=True)
# model.evaluate(x_test, y_test)
# model.save(cnn_model_path)



# ResNet50 training
# resnet_m = ResNet_Builder(input_shape, num_classes, metrics)
# resnet_m.summary()
# resnet_m.fit(x_train, y_train, batch_size, epochs, plot=True)
# resnet_m.evaluate(x_test, y_test)
# resnet_m.save(resnet_model_path)


# Inception training on pre trained weights
# inception_model = Inception_Builder(input_shape, num_classes, metrics)
# inception_model.summary()
# inception_model.fit(x_train, y_train, batch_size, epochs, plot=True)
# inception_model.evaluate(x_test, y_test)
# inception_model.save(inception_v3_path)


#MobilNet training on pre trained weights
mob_net_model = MobNet_Builder(input_shape, num_classes, metrics)
mob_net_model.summary()
mob_net_model.fit(x_train, y_train, batch_size, epochs, plot=True)
mob_net_model.evaluate(x_test, y_test)
mob_net_model.save(mobil_net_path)



# AUC and ROC graphs

KLASS = [0, 1]
y_pred_keras = mob_net_model.model.predict(x_test)
y_probs = []
for prob in y_pred_keras:
    if np.argmax(prob) == 0:
        y_probs.append(1-np.max(prob))
    else:
        y_probs.append(np.max(prob))


y_pred = [np.max(pair) for pair in y_probs]
y_test_ = [KLASS[np.argmax(pair)] for pair in y_test]
print(y_pred)
print(y_test_)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='MobileNet (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='MobileNet (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()








