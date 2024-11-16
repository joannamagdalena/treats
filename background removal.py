import glob
from rembg import remove
from PIL import Image

def choose_images_to_background_removal():
    training_data_path = "../training_data"
    dataset_files = glob.glob(training_data_path+"/*/*/*")
    prepared_training_data_path = "../training_data_without_background"
    prepared_dataset_files = glob.glob(prepared_training_data_path+"/*/*/*")

    training_files_diff = [f[len(training_data_path):] for f in dataset_files
                           if prepared_training_data_path+f[len(training_data_path):] not in prepared_dataset_files]

    return training_files_diff



choose_images_to_background_removal()