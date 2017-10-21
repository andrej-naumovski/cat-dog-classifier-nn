import cv2
import numpy as np
import random
import os


"""
@:param trimap - The image trimap
@:return alpha - The generated alpha
"""
def generate_alpha_from_trimap(trimap):
    alpha = trimap * 130 + 120  # should generate these values in a different manner, works for this dataset
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            for k in range(alpha.shape[2]):
                if alpha[i][j][k] == 124:
                    alpha[i][j][k] = 0

    return alpha.astype(float) / 255


def extract_object(image, trimap):
    alpha = generate_alpha_from_trimap(trimap=trimap)
    image = image.astype(float)

    image = cv2.multiply(alpha, image)
    return cv2.resize(image, (150, 200))


def load_dataset(dataset_txt, test_data=False):
    dataset_labels = np.loadtxt(dataset_txt, dtype='str')
    images = []
    labels = []
    dir = 'testobjects' if test_data else 'objects'
    if not os.path.isdir(dir):
        os.mkdir(dir)
    for i in range(dataset_labels.shape[0]):
        print('Getting image {}'.format(i + 1))
        print('{}/{}.jpg'.format(dir, dataset_labels[i][0]))
        label = [0, 0]
        label[int(dataset_labels[i][2]) - 1] = 1
        labels.append(label)
        if os.path.exists('{}/{}.jpg'.format(dir, dataset_labels[i][0])):
            image = cv2.imread('{}/{}.jpg'.format(dir, dataset_labels[i][0]))
            print(image)
        else:
            image = cv2.imread('images/{}.jpg'.format(dataset_labels[i][0]))
            trimap = cv2.imread('annotations/trimaps/{}.png'.format(dataset_labels[i][0]))
            image = extract_object(image, trimap=trimap)
        if not os.path.exists('{}/{}.jpg'.format(dir, dataset_labels[i][0])):
            cv2.imwrite('{}/{}.jpg'.format(dir, dataset_labels[i][0]), image)
        image = np.reshape(image, newshape=(image.size))
        print(image)
        images.append(image / 255)

    return np.array(images, dtype=np.float32), labels


def generate_batch(images, labels, batch_size):
    batch = {
        'images': [],
        'labels': []
    }

    for i in range(batch_size):
        index = random.randint(0, images.shape[0] - 1)
        batch['images'].append(images[index])
        batch['labels'].append(labels[index])

    return batch
