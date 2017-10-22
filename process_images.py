import image_utils
import cv2

images = image_utils.load_dataset(dataset_txt='annotations/trainval.txt')
images_test = image_utils.load_dataset(dataset_txt='annotations/test.txt', test_data=True)