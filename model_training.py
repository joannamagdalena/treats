from matplotlib import pyplot as plt
import glob
import tensorflow as tf


# loading picture data
def load_training_data():
    training_data_path = "./training_data"
    number_of_pictures = len(list(glob.glob('./training_data/*/*.jpg')))

    batch_size = int(number_of_pictures/3)
    pic_height = 180
    pic_width = 180

    training_dataset = tf.keras.utils.image_dataset_from_directory(
        training_data_path,
        image_size=(pic_height, pic_width),
        batch_size=batch_size
    )

    return training_dataset


# training the model
def train_model():
    # loading dataset
    training_dataset = load_training_data()
    class_names = training_dataset.class_names
    number_of_classes = len(class_names)

    autotune = tf.data.AUTOTUNE
    training_dataset = training_dataset.cache().prefetch(buffer_size=autotune)

    # normalization of training dataset
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_training_dataset = training_dataset.map(lambda x, y: (normalization_layer(x), y))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((180, 180, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(number_of_classes)]
    )

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])




#train_model()