# AI-Image-Recognition
A simple AI image recognition program, which can be easily modified to detect different things.
## How to install:
first clone this repository using:
```sh
git clone "https://github.com/PetrucciCarl/AI-Image-Recognition.git"
```
then, delete the .gitkeep file **(this is an important step)**:
```sh
del images/.gitkeep
```
then, install all required dependancies:
```sh
pip install -r requirements.txt
```
If you want GPU acceleration, install PyTorch manually:
https://pytorch.org/get-started/locally/
## How to train:
First you will need to gather a large collection of images for each different category you want you model to detect.
A larger amount of training data will produce a more accurate model.

Then, place these images in the 'images' folder within subfolders (e.g.if you want your model to detect 'hotdog' or 'not-hotdog', place the sorted images in folders with the names 'hotdog' and 'not-hotdog', if you want to check for different numbers, place images of numbers within folders labled from 0 to 9)

Then train the model by running:
```sh
python main.py train
```
This will default to training the model using 100 epochs. To change the number of epochs, you can add a number to your arguments, for instance:
```sh
python main.py train 1000
```
will train the model using 1000 epochs.

## How to run:
**You must train your model first before trying to run it**
Run the command:
```sh
python main.py run
```
You will then see a GUI appear, press the 'Select an image' button, to select an image file on your computer, the model will then run and try to identify which folder the image belongs to, based on the folders you setup duing training
