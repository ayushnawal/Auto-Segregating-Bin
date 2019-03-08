# Garbage Auto Segregating Bin
## Pytorch & Hardware implementation
An image classifying model segregator.ipynb will segregate the garbage into bio-degradable and non-biodegradable waste .
Using RaspberryPi and Servo Motor , we implemented the model by building a small dustbin , the top plate of which can flip +90 or -90 degrees depending on the category of waste .

![53615529_293853624626912_1072391129799852032_n](https://user-images.githubusercontent.com/36100944/54047292-ad813180-41fc-11e9-8bac-40e1b088364c.jpg)

Dataset we used contains 6 different classes of garbage :
[ cardboard , glass , metal , paper , plastic , trash]

Once the webcam detects a garbage on the plate , model segregates it using weights (model1.pt) and returns a final value . That value is passed to pi using scp(script.sh) and then servo motor responds(Servo.py) according to the value passed to it . 

## TensorFlow
Also tried segregating by building a tensorflow model , but the model was overfitting the data and as a result validation accuracy was only around 55 % . 
I will try to improve and update the model.

