# Face Tracking In Real Time
Implementation of a deep learning model based on the VGG16 neural network to track faces on video or with a camera in real time. This proyect has been done following [this](https://www.youtube.com/watch?v=N_W4EYtsa10&t=6473s) tutorial, in which [Nicholas Renotte](https://www.youtube.com/@NicholasRenotte) implemented a model using the Tensorflow framework, and I took it to deploy and port it to PyTorch. To run the implementation, download this repository and run the ```setup.py``` script. It will create the necessary folders to prepare the dataset.

IMPORTANT: The model is designed to detect and track faces in images/videos with a single face, so, in your dataset, put images with only one face. Also, the dataset I used was made with images in landscape orientation (e.g. 1280x720, 640x360). Finally, make sure your images are in _jpg_ format.


## Dataset preparation
To prepare the dataset you can use your own images (as I did) or you can connect a camera to the PC and run the ```capture_images_camera.py``` script to capture images. This script will capture and store the desired number of images (by default 100). If you choose the first option just put the images in the ```data/images``` folder, otherwise try to take images from different points of view moving your head and face. The script is programmed so that images are taken every 0.5 seconds, and they will be stored automatically in the folder. To set the number of images to take, you can change the default value in the script or open a terminal and run the next command:
```
python capture_images_camera.py --images_to_take=<number of images>
```
Use images with and without faces on them. Once you have your images, run the ```resize_images.py``` script to make them be the same size. By default the output width and height are set to 640 and 360 respectively. In case you want to change them (not recommended) run the next command:
```
python resize_images.py --width=<width> --height=<height>
```
The resized images will be stored in the ```data/images_resized``` folder.

The next step is to label those images. For that, the tool _Labelme_ will be used. Follow the tutorial mentioned above, starting at 20:05, to install the tool and label the images. Just note that when you click the _Open Dir_ option in the tool, the input directory in our case is ```data/images_resized```, and the output directory for the labels is ```data/labels```.

If the labeling process was successful, running ```visualize_dataset.py``` should display the images and the bounding boxes around the faces in the images.

## Data augmentation
To do the data augmentation, just run the ```data_augmentation.py``` script. For each image, using the library _albumentations_, one hundred images will be created taking subimages from the original image changing properties such as rotation, orientation, color, brightness... The resolution of these new images, i.e. their size, is set by deafutlt to 256x144. It can be changed by running the following command, but keep in mind that if this size is modified, the neural network and test files will also have to be modified, so it is not recommended. The number of subimages to take can be modified to (see the command).
```
python data_augmentation.py --width=<width> --height=<height> --num_subimages=<number of subimages>
```
The labels for those new images will be created automatically, setting the label to 1 or 0 depending on whether there is a face in the images or not, respectively, and converting the coordinates of the bounding boxes to the corresponding values. Images will be stored in the ```aug_data/images``` folder and labels will be stored in the ```aug_data/labels``` folder. To check that the data augmentation has been succesful, running ```visualize_augmented_dataset.py``` should display the new images with the bounding boxes around the faces.

## Train the model
To train the model, the convolutional neural network VGG16 has been used. If you followed the steps correctly, it should be enough to run the ```train.py``` script to start the training. The hyperparameters can be modified both setting them in the script or in the command line, and they are:
- ```dataset_path```: path to the fodler that contains the images and the labels. In our case '_aug_data_'.
- ```train_split```: percentage of the dataset to use for training.
- ```log_dir```: directory to store the checkpoints and loss values of the current training.
```
