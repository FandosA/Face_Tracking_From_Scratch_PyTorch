# Face Tracking In Real Time
Implementation of a deep learning model based on the VGG16 neural network to track faces on video or with a camera in real time. This proyect has been done following [this](https://www.youtube.com/watch?v=N_W4EYtsa10&t=6473s) tutorial, in which [Nicholas Renotte](https://www.youtube.com/@NicholasRenotte) implemented a model using the Tensorflow framework, and I took it to deploy and port it to PyTorch. To run the implementation, download this repository and run the ```setup.py``` script. It will create the necessary folders to prepare the dataset.

## Dataset preparation
To prepare the dataset you can use your own images (as I did) or you can connect a camera to the PC and run the ```capture_images_camera.py``` script to capture images. This script will capture and store the desired number of images. If you choose this second option try to take images from different points of view moving your head and face. Images will be captured every 0.5 seconds. To set the number of images, open a terminal and run the next command:
```
python capture_images_camera.py --images_to_take=<number of images>
```
Once you have your images, run the ```resize_images.py``` script to make images be the same size. By default the output width and height is set to 640 and 480 respectively. In case you want to change it run the next command:
```
python resize_images.py --width=<width> --height=<height>
```
