from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.metrics import mean_absolute_error
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout,\
                         Concatenate


def mae_months(mu, sigma):
    """
    loss function

    :param mu: mu
    :param sigma: sigma
    :return: loss
    """
    def loss(in_gt, in_pred):
        return mean_absolute_error(mu+sigma*in_gt, mu+sigma*in_pred)
    return loss


def get_bone_gender_age_vgg_model(image_shape, logger, mu, sigma):
    """
    Load data from training dataframe

    :param image_shape: image_shape, (224, 224)
    :param logger: logger
    :param mu: mu
    :param sigma: sigma
    :return: bone_age_model
    """

    logger.info("=== Start to compile model ===")
    img = Input(shape=image_shape)
    gender = Input(shape=(1,))
    vgg_model = VGG16(input_shape=image_shape,
                      include_top=False,
                      weights='imagenet')(img)
    cnn_vec = GlobalAveragePooling2D()(vgg_model)
    cnn_vec = Dropout(0.2)(cnn_vec)
    gender_vec = Dense(32, activation='relu')(gender)
    features = Concatenate(axis=-1)([cnn_vec, gender_vec])
    dense_layer = Dense(1024, activation='relu')(features)
    dense_layer = Dropout(0.2)(dense_layer)
    dense_layer = Dense(1024, activation='relu')(dense_layer)
    dense_layer = Dropout(0.2)(dense_layer)
    output_layer = Dense(1, activation='linear')(dense_layer)
    bone_age_model = Model(inputs=[img, gender], outputs=output_layer)
    for layer in bone_age_model.layers[0:2]:
        layer.trainable = False
    # Check the trainable status of the individual layers
    for layer in bone_age_model.layers:
        print(layer, layer.trainable)
    # Compile model
    bone_age_model.compile(optimizer='adam',
                           loss='mse',
                           metrics=[mae_months(mu=mu, sigma=sigma)])
    bone_age_model.summary(print_fn=logger.info)
    logger.info("=== Model compiled ===")

    return bone_age_model
