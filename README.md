# shoes_design
Generate new shoes images with Google Colaboratory

# Model
DCGAN - Deep Convolutional Generative Adversarial Networks


# Requirements
- Subscription on Google Colaboratory with Google Drive account =>  GPU mode on the Google Colaboratory Notebook
- pip install torchnet
- prepared data with splited folders in one folder : see 1.zip folder 
- unprepared data : source: http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ see: "ut-zap50k-images-square/Shoes/Sneakers and Athletic Shoes" folder

# Hyperparameters
You can use this model with different hyperparameters values.
![image](https://user-images.githubusercontent.com/46371678/101190566-d2683a00-3658-11eb-807d-2323a81f3fb2.png)

Here I put the DCGAN4_mod version because the resulted fake image looks better with the loaded weights model :
![image](https://user-images.githubusercontent.com/46371678/101190779-27a44b80-3659-11eb-990f-3e578621ff7f.png)

# What's new ?
The script is founded in : https://github.com/hminle/shoe-design-using-generative-adversarial-networks

I added some changes :
- to save, load and use weights in the model G
- to display fake and real images
- to display a video for the training process

# Comment
These models do not take too much time in the training process, about 20 minutes. 
But the images are too small we can't even see the shoes details. One image is about 3 Ko.
With a bigger picture the model could take much more time to run. That is another issue to look at.
