import image_utils

images = image_utils.load_dataset(dataset_txt='annotations/trainval.txt')
images_test = image_utils.load_dataset(dataset_txt='annotations/test.txt', test_data=True)