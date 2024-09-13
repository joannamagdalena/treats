from picture_analysis import *
from matplotlib import pyplot as plt


def choose_treat_type(animal):
    treat_dict = {"dog": "dog_treat", "cat": "cat_treat", "bird": "bird_treat", "error": "no treat", "other": "no treat"}
    return treat_dict[animal]


def choose_treat(picture):
    animal = check_animal(picture)
    return choose_treat_type(animal)


picture_name = input()
picture_input = plt.imread(picture_name)
print(choose_treat(picture_input))

