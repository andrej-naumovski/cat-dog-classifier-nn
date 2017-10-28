# CNN dog and cat classifier

Written in TensorFlow. 3 convolutional layers ran at 80% accuracy. Haven't been able to test with 8 layers due to hardware limitations.

Uses the [Oxford IIIT - Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/).

Scales all the images to 150x200x3, then uses the provided trimap to extract only the object. Image processing done using OpenCV.
