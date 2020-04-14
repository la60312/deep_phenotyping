from lib.data import *
from lib.model import get_bone_gender_age_vgg_model
from lib.visualize import draw_distribution_comparison
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as sk_mae
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import os
import time
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


def get_logger():
    """
    setup logger

    :return: logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join('run.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s: %(name)s - %(message)s',
                        datefmt='%m-%d %H:%M',
                        level=logging.DEBUG,
                        filemode='w')
    console_handle = logging.StreamHandler()
    console_handle.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s: %(message)s',
                                  datefmt='%m-%d %H:%M')
    console_handle.setFormatter(formatter)
    logger.addHandler(console_handle)

    # work around for the font warning in server
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    return logger


def load_data_from_dataframe(logger, data_dir='data/rsna-bone-age'):
    """
    Load data from training dataframe

    :param logger: logger
    :param data_dir: path to dataset, default: rsna-bone-age
    :return: age_df
    """
    img_dir = os.path.join(data_dir,
                           "boneage-training-dataset",
                           "boneage-training-dataset")
    csv_path = os.path.join(data_dir, "boneage-training-dataset.csv")
    logger.info('Loading training metadata from {}'.format(csv_path))
    df = pd.read_csv(csv_path)

    df['path'] = df['id'].map(lambda x: os.path.join(img_dir, "{}.png".format(x)))
    df['exists'] = df['path'].map(os.path.exists)
    df['gender'] = df['male'].map(lambda x: "male" if x else "female")
    mu = df['boneage'].mean()
    sigma = df['boneage'].std()
    df['zscore'] = df['boneage'].map(lambda x: (x-mu)/sigma)
    df.dropna(inplace=True)

    logger.info('Total image in training data: {}'.format(df.shape[0]))
    logger.info('Number of male: {}'.format(df[df['male']==True].shape[0]))
    logger.info('Number of female: {}'.format(df[df['male']==False].shape[0]))

    # show statistic
    dis_1 = df[df['male']==True][['boneage']].values.ravel()
    dis_2 = df[df['male']==False][['boneage']].values.ravel()
    draw_distribution_comparison(dis_1, dis_2,
                                 'gender_bone_age_distribution.png',
                                 logger,
                                 bin_size=20, xlabel='bone age',
                                 title='Bone age distribution',
                                 label_1='Male', label_2='Female')

    dis_1 = df[df['male']==True][['male']].values.ravel()
    dis_2 = df[df['male']==False][['male']].values.ravel()
    draw_distribution_comparison(dis_1, dis_2, 'gender_distribution.png',
                                 logger,
                                 bin_size=20, xlabel='gender',
                                 title='Gender distribution',
                                 label_1='Male', label_2='Female')
    df['boneage_category'] = pd.cut(df['boneage'], 10)

    return df, mu, sigma


def split_dataset(df, logger, seed=0):
    """
    Split dataset into training, validation and testing

    :param df, dataframe
    :param logger: logger
    :param seed: random seed, default: 0
    :return: train_df, valid_df, test_df
    """

    logger.info("Preparing training, testing and validation datasets ...")
    df['boneage_category'] = pd.cut(df['boneage'], 10)
    raw_train_df, test_df = train_test_split(df,
                                             test_size=0.2,
                                             random_state=seed,
                                             stratify=df['boneage_category'])
    raw_train_df, valid_df = train_test_split(raw_train_df,
                                              test_size=0.1,
                                              random_state=seed,
                                              stratify=raw_train_df['boneage_category'])

    # Balance the distribution in the training set
    # We should draw some figure to validate this
    train_df = raw_train_df.groupby(['boneage_category', 'male'])\
        .apply(lambda x: x.sample(500, replace=True)).reset_index(drop=True)

    return train_df, valid_df, test_df


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    start = time.time()
    RAND_SEED = 2408
    IMG_SIZE = (224, 224)
    logger = get_logger()
    logger.info("=== Start bone age prediction ===")

    # Load metadata from csv
    df, mu, sigma = load_data_from_dataframe(logger)

    # Split into training testing and validation datasets
    train_df, valid_df, test_df = split_dataset(df, logger, seed=RAND_SEED)
    train_size = train_df.shape[0]
    valid_size = valid_df.shape[0]
    test_size = test_df.shape[0]
    logger.info("Training images:   {}".format(train_size))
    logger.info("Validation images: {}".format(valid_size))
    logger.info("Testing images:    {}".format(test_size))

    img_gen = get_img_data_gen()
    train_gen = flow_from_dataframe(img_gen,
                                    train_df,
                                    path_col='path',
                                    y_col='zscore',
                                    gender_col='male',
                                    batch_size=32,
                                    seed=RAND_SEED,
                                    img_size=IMG_SIZE)

    logger.info("Preparing validation data...")
    # Get the validation data
    valid_gen = flow_from_dataframe(img_gen,
                                    valid_df,
                                    path_col='path',
                                    y_col='zscore',
                                    gender_col='male',
                                    batch_size=valid_size,
                                    seed=RAND_SEED,
                                    img_size=IMG_SIZE)
    valid_X, valid_Y = next(valid_gen)
    IMG_SHAPE = valid_X[0][0, :, :, :].shape
    logger.info("Image shape: "+str(IMG_SHAPE))
    logger.info("Data preproation done")

    # Model definition
    bone_age_model = get_bone_gender_age_vgg_model(IMG_SHAPE,
                                                   logger,
                                                   mu,
                                                   sigma)

    logger.info("=== Star training model ===")
    # Model Callbacks
    epochs = 20
    weight_path = "bone_age_weights_untrainable_VGG16_gender" +\
                  "_{}_epochs_relu_less_dropout_dense.best.hdf5".format(epochs)
    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.8,
                                       patience=10,
                                       verbose=1,
                                       mode='auto',
                                       epsilon=0.0001,
                                       cooldown=5,
                                       min_lr=0.0001)

    early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    callbacks_list = [checkpoint, early, reduceLROnPlat]
    if not os.path.exists(weight_path):
        history = \
            bone_age_model.fit_generator(train_gen,
                                         steps_per_epoch=train_size/32,
                                         validation_data=(valid_X, valid_Y),
                                         epochs=epochs,
                                         callbacks=callbacks_list,
                                         verbose=1)
        with open('history_gender_vgg16_freeze_epoch_{}.p'.format(epochs), 'wb') as f:
            pickle.dump(history.history, f)
    bone_age_model.load_weights(weight_path)
    logger.info("Training complete !!!\n")

    # Evaluate model on test dataset
    logger.info("Evaluating model on test data ...\n")
    logger.info("Preparing testing dataset...")
    test_gen = flow_from_dataframe(img_gen,
                                   test_df,
                                   path_col='path',
                                   y_col='zscore',
                                   gender_col='male',
                                   batch_size=test_size,
                                   seed=8309,
                                   img_size=IMG_SIZE)
    test_X, test_Y = next(test_gen)
    logger.info("Data prepared !!!")

    pred_Y = mu+sigma*bone_age_model.predict(x=test_X,
                                             batch_size=25,
                                             verbose=1)
    test_Y_months = mu+sigma*test_Y
    logger.info("Mean absolute error on test data: "
                + str(sk_mae(test_Y_months, pred_Y)))

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    ax1.plot(test_Y_months, pred_Y, 'r.', label='predictions')
    ax1.plot(test_Y_months, test_Y_months, 'b-', label='actual')
    ax1.legend()
    ax1.set_xlabel('Actual Age (Months)')
    ax1.set_ylabel('Predicted Age (Months)')
    plt.savefig('prediction_gender_epoch_{}.png'.format(epochs))

    ord_idx = np.argsort(test_Y)
    ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, num=8).astype(int)]
    fig, m_axs = plt.subplots(2, 4, figsize=(16, 32))
    for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):
        c_ax.imshow(test_X[0][idx, :, :, 0], cmap='bone')
        title = 'Age: %2.1f\nPredicted Age: %2.1f\nGender: ' % (test_Y_months[idx], pred_Y[idx])
        if test_X[1][idx] == 0:
            title += "Female\n"
        else:
            title += "Male\n"
        c_ax.set_title(title)
        c_ax.axis('off')
    plt.savefig('visulize_xray.png')
    # Done
    total_sec = time.time() - start
    logger.info("Total run took {} (Hours:Min:Sec)".format(str(datetime.timedelta(
        seconds=total_sec))))
    logger.info("done!")


if __name__ == '__main__':
    main()
