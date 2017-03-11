# neural_art

This repository contains (TensorFlow and Keras) code that transfers the 
style of a given image to another. The code has been written using [this][original] and [this][youtube] as a guide

## Input image 

<img src="/logan.jpg" width="256" height="256">

## Style Images
<img src="/styles/wave.jpg" width="256" height="256">
<img src="/styles/forest.jpg" width="256" height="256">


## Resultant Images

<img src="/result.png" width="600" height="400">

## Final Image
This image was generated after the final iteration

<img src="/output/result9.png" width="256" height="256">

## Run

To run the script just open the terminal and change directory to
this repository.
Run this python file **artist.py**


## Remarks

The code runs in 10 iterations.
After every iteration an Image is generated and saved in the output folder.
Every iteration takes upto 20 mins since its working on a **CPU** and not **GPU**.
So if you have NVIDIA GPU on your machine then it is recommended that you use 
Tensorflow with GPU support for faster execution. 



[original]: https://github.com/hnarayanan/artistic-style-transfer/blob/master/notebooks/6_Artistic_style_transfer_with_a_repurposed_VGG_Net_16.ipynb

[youtube]: https://www.youtube.com/watch?v=Oex0eWoU7AQ&feature=youtu.be