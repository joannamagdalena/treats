from matplotlib import pyplot as plt
import tensorflow as tf
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def prepare_training_data(dataset, shuffle=False, augment=False):
    # resizing and normalization of dataset
    resizing_layer = tf.keras.layers.Resizing(180, 180)
    dataset = dataset.map(lambda x, y: (resizing_layer(x), y))
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

    if shuffle:
        dataset = dataset.shuffle(10)

    # data augmentation
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2)
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# loading picture data
def load_training_data(animal_type):
    training_data_path = "./training_data/"

    batch_size = 1
    pic_height = 180
    pic_width = 180

    training_dataset = tf.keras.utils.image_dataset_from_directory(
        training_data_path+animal_type,
        validation_split=0.25,
        subset="training",
        seed=123,
        image_size=(pic_height, pic_width),
        batch_size=batch_size
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        training_data_path+animal_type,
        validation_split=0.25,
        subset="validation",
        seed=123,
        image_size=(pic_height, pic_width),
        batch_size=batch_size
    )

    class_names = training_dataset.class_names
    training_dataset = prepare_training_data(training_dataset, shuffle=True, augment=True)
    validation_dataset = prepare_training_data(validation_dataset, shuffle=True, augment=True)

    return training_dataset, validation_dataset, class_names


# training the model
def train_model(animal_type):
    # loading dataset
    training_dataset, validation_dataset, class_names  = load_training_data(animal_type)
    number_of_classes = len(class_names)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='softmax'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='softmax'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='softmax'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='softmax'),
        tf.keras.layers.Dense(number_of_classes)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )

    epochs_number = 10
    history = model.fit(training_dataset, validation_data=validation_dataset, epochs=epochs_number)
    acc = history.history['accuracy']
    val_acc = history.history["val_accuracy"]
    loss = history.history['loss']
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs_number), acc, label='Training Accuracy')
    plt.plot(range(epochs_number), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs_number), loss, label='Training Loss')
    plt.plot(range(epochs_number), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training Loss')
    plt.show()

    return model, class_names

#train_model()
#load_training_data("cat")