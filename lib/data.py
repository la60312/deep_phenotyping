import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.applications.vgg16 import preprocess_input 

def doubleGenerator(genX1, genX2):
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[1]], X1i[1]  #Yield both images and their mutual label


def datagen(in_df, path_col, y_col, batch_size, **dflow_args):
    img_data_gen = ImageDataGenerator(samplewise_center=False, 
            samplewise_std_normalization=False, 
            horizontal_flip = True, 
            vertical_flip = False, 
            height_shift_range = 0.15, 
            width_shift_range = 0.15, 
            rotation_range = 5, 
            shear_range = 0.01,
            fill_mode = 'reflect',
            zoom_range=0.25,
            preprocessing_function = preprocess_input)

    base_dir = os.path.dirname(in_df[path_col].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir,
            class_mode='sparse',
            batch_size = batch_size,
            shuffle=True, **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.classes = np.stack(in_df[y_col].values)
    return df_gen

def get_img_data_gen():
    img_data_gen = ImageDataGenerator(samplewise_center=False,
                                      samplewise_std_normalization=False,
                                      horizontal_flip=True,
                                      vertical_flip=False,
                                      height_shift_range=0.15,
                                      width_shift_range=0.15,
                                      rotation_range=5,
                                      shear_range=0.01,
                                      fill_mode='reflect',
                                      zoom_range=0.25,
                                      preprocessing_function=preprocess_input)
    return img_data_gen


def flow_from_dataframe(imgDatGen, df, path_col, y_col, gender_col, batch_size, seed, img_size):
    gen_img = imgDatGen.flow_from_dataframe(dataframe=df,
            x_col=path_col, y_col=[y_col, gender_col],
            batch_size=batch_size, seed=seed, shuffle=True, class_mode='multi_output',
            target_size=img_size, color_mode='rgb')

    while True:
        X1i = gen_img.next()
        yield [X1i[0], X1i[1][1]], X1i[1][0]
