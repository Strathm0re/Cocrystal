Getting started

Prerequisites
Linux 
PyTorch
NVIDIA GPU

Installation
The code requires python=3.10, as well as torch=2.4.0 and torchvision=0.19.0. 
Please follow the instructions here to install both PyTorch and TorchVision dependencies.

Clone this repo:
git clone https://github.com/xxx.git
Install libraries:
pip install -r requirements.txt

Training and evaluation
Train ResNet-18 model
train.py
Test ResNet-18 model
test.py

ImagePrediction
The model checkpoint can be acquired from the folder named checkpoint.
image->type
image_path = file_dir + r"/sample/Crystal4.JPG"  
predicted_class, predicted_label = predict_image(image_path)
Please see prediction_img.ipynb for more details.


Reference
1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.
2. Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). Dive into Deep Learning. Cambridge University Press. https://D2L.ai

Contact
Feel free to contact us if there is any question:
for code xx@qq.com; for dataset xx@xx.com.
