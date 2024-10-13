from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch
from model_training import train_model
import tensorflow as tf
import numpy as np


def calculate_probabilities_for_classes(picture):
    animal_classes = ["dog", "cat", "bird", "other"]

    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", clean_up_tokenization_spaces=True)

    inputs = processor(images=picture, text=animal_classes, return_tensors="pt", padding=True)

    with torch.no_grad():
        output = model(**inputs)

    probabilities = output.logits_per_image[0].softmax(dim=-1).numpy()
    probabilities_list = list(probabilities)

    result = [{"probability": prob, "class": animal_class}
              for prob, animal_class in sorted(zip(probabilities_list, animal_classes), key=lambda x: -x[0])]

    return result


def check_animal(picture, picture_keras):
    animal_type = "error"

    probabilities_for_classes = calculate_probabilities_for_classes(picture)
    highest_probability = list(probabilities_for_classes[0].values())

    if highest_probability[0] >= 0.9:
        animal_type = highest_probability[1]
    elif highest_probability[1] == "cat" and highest_probability[0] >= 0.8:
        animal_type = "cat"

    model, animal_classes = train_model(animal_type)
    animal_prediction = model.predict(picture_keras)
    animal_prediction_class = animal_classes[np.argmax(animal_prediction[0])]
    animal_prediction_probability = np.max(tf.nn.softmax(animal_prediction[0])) * 100

    print(animal_prediction_probability)
    print(animal_prediction_class)

    return animal_type

