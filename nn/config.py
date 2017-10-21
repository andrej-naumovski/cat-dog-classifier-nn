layer1 = {
    'filter_size': 5,
    'num_filters': 16
}

layer2 = {
    'filter_size': 2,
    'num_filters': 36
}

layer3 = {
    'filter_size': 2,
    'num_filters': 48
}

fully_connected_size = 128


img_width = 150
img_height = 200
num_channels = 3
img_size_flat = img_width * img_height * num_channels
img_shape = (img_width, img_height)
num_classes = 2
