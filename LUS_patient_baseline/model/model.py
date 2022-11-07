from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.layers.convolutional import Convolution2D, Conv2DTranspose
from tensorflow.python.keras.layers.core import Activation, Reshape
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import UpSampling2D, MaxPooling2D, concatenate
from keras import backend as K


def dice(y_pred, y_true):
  y_true_f = K.flatten(y_true[:, :, :-1])
  y_pred_f = K.flatten(y_pred[:, :, :-1])
  # Try thresholding 0.5
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_loss(y_true, y_pred):
  return 1-dice(y_true, y_pred)


def fbeta(y_pred, y_true):

  pred0 = Lambda(lambda x: x[:, :, :, 0])(y_pred)
  pred1 = Lambda(lambda x: x[:, :, :, 1])(y_pred)
  true0 = Lambda(lambda x: x[:, :, :, 0])(y_true)
  true1 = Lambda(lambda x: x[:, :, :, 1])(y_true)  # channel last?

  y_pred_0 = K.flatten(pred0)
  y_true_0 = K.flatten(true0)

  y_pred_1 = K.flatten(pred1)
  y_true_1 = K.flatten(true1)

  intersection0 = K.sum(y_true_0 * y_pred_0)
  intersection1 = K.sum(y_true_1 * y_pred_1)

  precision0 = intersection0/(K.sum(y_pred_0)+K.epsilon())
  recall0 = intersection0/(K.sum(y_true_0)+K.epsilon())

  precision1 = intersection1/(K.sum(y_pred_1)+K.epsilon())
  recall1 = intersection1/(K.sum(y_true_1)+K.epsilon())

  fbeta0 = (1.0+0.25)*(precision0*recall0) / \
      (0.25*precision0+recall0+K.epsilon())
  fbeta1 = (1.0+4.0)*(precision1*recall1) / \
      (4.0*precision1+recall1+K.epsilon())

  return ((fbeta0+fbeta1)/2.0)


def fbeta_loss(y_true, y_pred):
  return 1-fbeta(y_true, y_pred)


def weighted_categorical_crossentropy(y_true, y_pred):
  # weights = K.variable([4.0, 4.0, 2.0, 0.5])
  weights = K.variable([0.0, 1.0])
  # scale predictions so that the class prob of each sample sum to 1
  y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
  # clip to prevent NaN's and Inf's
  y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
  # calc
  loss = y_true * K.log(y_pred) * weights
  loss = -K.sum(loss, -1)
  return loss


def cat_dice_loss(y_true, y_pred):
  return weighted_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
  # return dice_loss(y_true, y_pred)


def segnet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax", model_number=1):

  if (model_number == 1):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(8, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_1 = Convolution2D(8, (kernel, kernel), padding="same")(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    pool_1 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_1)

    conv_2 = Convolution2D(16, (kernel, kernel), padding="same")(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    conv_2 = Convolution2D(16, (kernel, kernel), padding="same")(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    pool_2 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_2)

    conv_3 = Convolution2D(32, (kernel, kernel), padding="same")(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_3 = Convolution2D(32, (kernel, kernel), padding="same")(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    pool_3 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_3)

    conv_4 = Convolution2D(64, (kernel, kernel), padding="same")(pool_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    conv_4 = Convolution2D(64, (kernel, kernel), padding="same")(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    pool_4 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_4)

    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(pool_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(conv_5)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)

    print("Encoder built..")

    # decoder
    unpool_6 = UpSampling2D(size=pool_size)(conv_5)
    conv_6 = Convolution2D(64, (kernel, kernel), padding="same")(
        concatenate([conv_4, unpool_6]))
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_6 = Convolution2D(64, (kernel, kernel), padding="same")(conv_6)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)

    unpool_7 = UpSampling2D(size=pool_size)(conv_6)
    conv_7 = Convolution2D(32, (kernel, kernel), padding="same")(
        concatenate([conv_3, unpool_7]))
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    conv_7 = Convolution2D(32, (kernel, kernel), padding="same")(conv_7)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    unpool_8 = UpSampling2D(size=pool_size)(conv_7)
    conv_8 = Convolution2D(16, (kernel, kernel), padding="same")(
        concatenate([conv_2, unpool_8]))
    conv_8 = BatchNormalization()(conv_8)
    unpool_8 = UpSampling2D(size=pool_size)(conv_8)
    conv_8 = Convolution2D(16, (kernel, kernel), padding="same")(conv_8)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)

    unpool_9 = UpSampling2D(size=pool_size)(conv_8)
    conv_9 = Convolution2D(8, (kernel, kernel), padding="same")(
        concatenate([conv_1, unpool_9]))
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_9 = Convolution2D(8, (kernel, kernel), padding="same")(conv_9)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)

    conv_10 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Reshape(
        (input_shape[0] * input_shape[1], n_labels),
        input_shape=(input_shape[0], input_shape[1], n_labels),
    )(conv_10)

    outputs = Activation(output_mode)(conv_10)
    print("Decoder built..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

  elif (model_number == 2):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(8, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    pool_1 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_1)

    conv_2 = Convolution2D(16, (kernel, kernel), padding="same")(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    pool_2 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_2)

    conv_3 = Convolution2D(32, (kernel, kernel), padding="same")(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    pool_3 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_3)

    conv_4 = Convolution2D(64, (kernel, kernel), padding="same")(pool_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    pool_4 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_4)

    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(pool_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)

    print("Encoder built..")

    # decoder
    unpool_6 = UpSampling2D(size=pool_size)(conv_5)
    conv_6 = Convolution2D(64, (kernel, kernel), padding="same")(
        concatenate([conv_4, unpool_6]))
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)

    unpool_7 = UpSampling2D(size=pool_size)(conv_6)
    conv_7 = Convolution2D(32, (kernel, kernel), padding="same")(
        concatenate([conv_3, unpool_7]))
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    unpool_8 = UpSampling2D(size=pool_size)(conv_7)
    conv_8 = Convolution2D(16, (kernel, kernel), padding="same")(
        concatenate([conv_2, unpool_8]))
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)

    unpool_9 = UpSampling2D(size=pool_size)(conv_8)
    conv_9 = Convolution2D(8, (kernel, kernel), padding="same")(
        concatenate([conv_1, unpool_9]))
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)

    conv_10 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Reshape(
        (input_shape[0] * input_shape[1], n_labels),
        input_shape=(input_shape[0], input_shape[1], n_labels),
    )(conv_10)

    outputs = Activation(output_mode)(conv_10)
    print("Decoder built..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

  elif (model_number == 3):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(8, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(16, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    conv_3 = Convolution2D(32, (kernel, kernel), padding="same")(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    # pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_3)
    pool_1 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_3)
    conv_4 = Convolution2D(64, (kernel, kernel), padding="same")(pool_1)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(conv_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)

    # pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_6)
    pool_2 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_6)
    print("Encoder built..")

    # decoder

    # unpool_1 = MaxUnpooling2D(pool_size)([pool_2, mask_2])
    unpool_1 = UpSampling2D(size=pool_size)(pool_2)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_1)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    conv_8 = Convolution2D(128, (kernel, kernel), padding="same")(conv_7)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(64, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)

    # unpool_2 = MaxUnpooling2D(pool_size)([conv_9, mask_1])
    unpool_2 = UpSampling2D(size=pool_size)(conv_9)
    conv_10 = Convolution2D(32, (kernel, kernel), padding="same")(unpool_2)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)
    conv_11 = Convolution2D(16, (kernel, kernel), padding="same")(conv_10)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)

    conv_12 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Reshape(
        (input_shape[0] * input_shape[1], n_labels),
        input_shape=(input_shape[0], input_shape[1], n_labels),
    )(conv_12)

    outputs = Activation(output_mode)(conv_12)
    print("Decoder built..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

  elif (model_number == 4):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(8, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_1 = Convolution2D(8, (kernel, kernel), padding="same")(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    pool_1 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_1)

    conv_2 = Convolution2D(16, (kernel, kernel), padding="same")(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    conv_2 = Convolution2D(16, (kernel, kernel), padding="same")(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    pool_2 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_2)

    conv_3 = Convolution2D(32, (kernel, kernel), padding="same")(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_3 = Convolution2D(32, (kernel, kernel), padding="same")(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    pool_3 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_3)

    conv_4 = Convolution2D(64, (kernel, kernel), padding="same")(pool_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    conv_4 = Convolution2D(64, (kernel, kernel), padding="same")(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    pool_4 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_4)

    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(pool_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(conv_5)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)

    print("Encoder built..")

    # decoder
    unpool_6 = Conv2DTranspose(
        64, (kernel-1, kernel-1), strides=(2, 2))(conv_5)
    conv_6 = Convolution2D(64, (kernel, kernel), padding="same")(
        concatenate([conv_4, unpool_6]))
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_6 = Convolution2D(64, (kernel, kernel), padding="same")(conv_6)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)

    unpool_7 = Conv2DTranspose(
        32, (kernel-1, kernel-1), strides=(2, 2))(conv_6)
    conv_7 = Convolution2D(32, (kernel, kernel), padding="same")(
        concatenate([conv_3, unpool_7]))
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    conv_7 = Convolution2D(32, (kernel, kernel), padding="same")(conv_7)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    unpool_8 = Conv2DTranspose(
        16, (kernel-1, kernel-1), strides=(2, 2))(conv_7)
    conv_8 = Convolution2D(16, (kernel, kernel), padding="same")(
        concatenate([conv_2, unpool_8]))
    conv_8 = BatchNormalization()(conv_8)
    unpool_8 = UpSampling2D(size=pool_size)(conv_8)
    conv_8 = Convolution2D(16, (kernel, kernel), padding="same")(conv_8)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)

    unpool_9 = Conv2DTranspose(
        8, (kernel-1, kernel-1), strides=(2, 2))(conv_8)
    conv_9 = Convolution2D(8, (kernel, kernel), padding="same")(
        concatenate([conv_1, unpool_9]))
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_9 = Convolution2D(8, (kernel, kernel), padding="same")(conv_9)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)

    conv_10 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Reshape(
        (input_shape[0] * input_shape[1], n_labels),
        input_shape=(input_shape[0], input_shape[1], n_labels),
    )(conv_10)

    outputs = Activation(output_mode)(conv_10)
    print("Decoder built..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

  return model
