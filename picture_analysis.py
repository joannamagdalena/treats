from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch


def calculate_probabilities_for_classes(picture):
    animal_classes = ["dog", "cat", "bird", "other"]

    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    inputs = processor(images=picture, text=animal_classes, return_tensors="pt", padding=True)

    with torch.no_grad():
        output = model(**inputs)

    probabilities = output.logits_per_image[0].softmax(dim=-1).numpy()
    probabilities_list = list(probabilities)

    result = [{"probability": prob, "class": animal_class}
              for prob, animal_class in sorted(zip(probabilities_list, animal_classes), key=lambda x: -x[0])]

    return result


def check_animal(picture):
    animal = "error"

    probabilities_for_classes = calculate_probabilities_for_classes(picture)
    highest_probability = list(probabilities_for_classes[0].values())

    if highest_probability[0] >= 0.9:
        animal = highest_probability[1]

    return animal

