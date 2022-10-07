import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

"""Define the base model.

To be compatible with TFLite Model Personalization, we need to define a
base model and a head model. 

Here we are using an identity layer for base model, which just passes the 
input as it is to the head model.
"""
base = tf.keras.Sequential(
    [tf.keras.Input(shape=(32, 32, 3)), tf.keras.layers.Lambda(lambda x: x)]
)

base.compile(loss="categorical_crossentropy", optimizer="sgd")
base.save("identity_model", save_format="tf")

"""Define the head model.

This is the model architecture that we will train using Flower. 
"""
head = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax"
)

head.compile(loss="categorical_crossentropy", optimizer="sgd")


"""Convert the model for TFLite.

Using 10 classes in CIFAR10, learning rate = 1e-3 and batch size = 32

This will generate a directory called tflite_model with five tflite models.
Copy them in your Android code under the assets/model directory.
"""

base_path = bases.saved_model_base.SavedModelBase("identity_model")
converter = TFLiteTransferConverter(
    1280, base_path, heads.KerasModelHead(head), optimizers.SGD(1e-3), train_batch_size=1
)

converter.convert_and_save("tflite_model")
