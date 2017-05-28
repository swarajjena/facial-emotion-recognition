# Emotion Recognition using Deep Learning
A  recognition system through webcam streaming/image input and CNN.

## Abstraction
This project aims to recognize facial expression with CNN implemented by Keras. 
<br />I also implement a real-time module which can real-time capture user's face through webcam steaming called by opencv. 
<br />OpenCV cropped the face it detects from the original frames and resize the cropped images to 48x48 grayscale image, 
<br />Then take them as inputs of deep leanring model. 

## Dataset
[fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) is the dataset I chose, which is anounced in Kaggle competition in 2013.


### Requirements
<br />NumPy
<br />Tensorflow r1.1
<br />Keras 2
<br />OpenCV2

## Usage
```
python predict.py -p sample.jpg
```


Please check Ipython notebook files for more details 
