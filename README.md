# Face Tracking In Real Time
Implementation of a deep learning model based on the VGG16 neural network to track faces on video or with a camera in real time. This proyect has been done following [this](https://www.youtube.com/watch?v=N_W4EYtsa10&t=6473s) tutorial, in which [Nicholas Renotte](https://www.youtube.com/@NicholasRenotte) implemented a model using the Tensorflow framework, and I took it to deploy and port it to PyTorch. To run the implementation, download this repository and run the ```setup.py``` script. It will create the necessary folders to prepare the dataset.

IMPORTANT: The model is designed to detect and track images/videos with a single face, so, in your dataset, put images with only one face.


## Dataset preparation
To prepare the dataset you can use your own images (as I did) or you can connect a camera to the PC and run the ```capture_images_camera.py``` script to capture images. This script will capture and store the desired number of images. If you choose the first option just put the images in the ```data/images``` folder, otherwise try to take images from different points of view moving your head and face (images will be stored automatically in the folder). Images are captured every 0.5 seconds. To set the number of images totake, you can change the default value in the script or open a terminal and run the next command:
```
python capture_images_camera.py --images_to_take=<number of images>
```
Use images with and without faces on them. Once you have your images, run the ```resize_images.py``` script to make them be the same size. By default the output width and height are set to 640 and 480 respectively. In case you want to change them (not recommended) run the next command:
```
python resize_images.py --width=<width> --height=<height>
```
The resized images will be stored in the ```data/images_resized``` folder.

The next step is to label those images. For that, the tool _Labelme_ will be used. Follow the tutorial mentioned above, starting at 20:05, to install the tool and label the images. Just note that when you click the _Open Dir_ option in the tool, the input directory in our case is ```data/images_resized```, and the output directory for the labels is ```data/labels```.

If the labeling process was successful, running ```visualize_dataset.py``` should display the dataset with the rectangles around the faces in the images.
