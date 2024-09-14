import logging
from datetime import date, datetime
from picture_analysis import *
from matplotlib import pyplot as plt

# logging
logger = logging.getLogger()
logging.basicConfig(filename=str(date.today()) + "-log.log", encoding="utf-8", level=logging.INFO)
logger.info(str(datetime.now()) + ": Started")


def choose_treat_type(animal):
    treat_dict = {"dog": "dog treat", "cat": "cat treat", "bird": "bird treat", "error": "no treat", "other": "no treat"}
    return treat_dict[animal]


def update_log_file(animal, treat):
    if animal in ["error", "other"]:
        logger.error(str(datetime.now()) + ": Couldn't identify the animal.")
    else:
        logger.info(str(datetime.now()) + ": A " + animal + " got some " + treat + ".")
    return True


def choose_treat(picture):
    animal = check_animal(picture)
    treat = choose_treat_type(animal)

    update_log_file(animal, treat)

    return treat


picture_name = input()  # picture file
try:
    picture_input = plt.imread(picture_name)
    choose_treat(picture_input)
except OSError:
    logger.error(str(datetime.now()) + ": No picture uploaded.")

logger.info(str(datetime.now()) + ": Finished")

