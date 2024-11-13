import glob
from rembg import remove
from PIL import Image

def choose_images_to_background_removal():
    training_data_path = "../training_data/"
    dataset_files = glob.glob(training_data_path+"/*/*/*")
    prepared_training_data_path = "../training_data_without_background/"
    prepared_dataset_files = glob.glob(prepared_training_data_path+"/*/*/*")


choose_images_to_background_removal()