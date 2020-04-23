# Dog Breed Detection
## Project Overview

A Convolutional Neural Network(CNN) model that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

## Project Motivation
The motivation for this project is solely for educational and entertainment activities. The aim is to use CNN Transfer Learning with Keras. The notebook is considered to be deployed as a web application in the nearest future.

## Problem Statement
The problem to be explored is the detection of breed. Convolutional Neural Networks are used to achieve this result. Expected result is the name of the breed returned after inputing an image (if there is a dog or human). If the photo does not include neither human nor dog, the user will be asked to upload another photo.

Sample output:

![Sample human output](https://i.ibb.co/82yzc4J/sample-human-output.png)
![Sample dog output](https://i.ibb.co/2n5d7vC/sample-dog-output.png)

## Metrics
To train the model, categorical crossentropy is used as the output is a multi-class classification. The performance of the model is evaluated by looking at test accuracy as this metric can accurately show how the model is doing. Considering that there is no imbalance in the dataset, this metric will not pose any problem.

## Analysis
There are 8351 total dog images in 133 dog categories. Training set includes 6680 images, validation - 835 images and test 836 images.
Face Detection Algorithm detects human face in 11% of 100 sample dog images.
No human images are detected as dogs by Dog Detection Algorithm.

## Mehodology
The data is normalized by dividing all the pixels by 255.
CNN without transfer learning showed the accuracy of 3%, which is extremely low. However, as this was done for only 5 epochs, the result could have been better if trained more. This step involved three convolutions and max poolings followed by global average pooling.
Transfer learning was used to obtained the most exact answers. Two pre-trained models were tested.
VGG-16 resulted in test accuracy of 44.85%, while InceptionV3 correctly classified 83% of the images. Therefore, it was decided to continue with InceptionV3.

## Results
The elected InceptionV3 model along with additional Global Average Pooling layer correctly classified 83% of the test images. Considering that some images are very confusing even for human eye, this accuracy is better than expected.
Predictions were made for some images found on the internet and personal ones. All of them were correct, which proves the viability of the model.

## Conclusion
The project showed how effective transfer learning can be in different classification problems, namely dog breed classification. The problem of detecting dog breed is solved by using a pre-trained InceptionV3 model and adding an additional layer. The results are impressive, as the model correctly classifies in more than 80% of cases.
However, some improvements could be done:
1. Currently, speed of the algorithm is not very optimal. This could have been tackled in more detail.
2. Other pre-trained models could be tested. Maybe some other model would perform better.
3. Hyperparameter tuning could have been done to identify the optimum parameters.

## Instructions to Run the Project

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/fidanmusazade/dog-app.git
cd dog-app
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. 
- Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

- Donwload the [Inception-V3 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.


5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

## License and Acknowledgments
License can be found [here](LICENSE). The project is accomplished with the help of Udacity.
