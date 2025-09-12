import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torchvision
from PIL import Image
from PIL import ImageEnhance
from d2l import torch as d2l
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import train
import read_dataset
import model.model as model
# -------------------------set parameters--------------------------
file_dir = r"/home/dell/Research/Disk/1_dataset/Classification_3/upload/"
data_dir = file_dir + r"/dataset/"
save_dir = file_dir + r"/Powder_SingleCrystal_Liquid_ResNet18/save_pth/test_save/"
checkpoint_path = file_dir + r"/Powder_SingleCrystal_Liquid_ResNet18/save_pth/train_save/1retrain_resnet18_batchsize_50_epoch_150_lp_1_ld_0.99.pth"
devices = d2l.try_all_gpus()
# -------------------------import test set--------------------------
loader = read_dataset.reader(data_dir, train.batch_size)
test_ds, test_iter = loader.test_ds, loader.test_iter
# ----------------------------import net---------------------------
# When using torch.nn.DataParallel during model training, the saved model will include a 'module.' prefix. 
# When loading the model, it is necessary to remove these prefixes.
net = model.get_net(pretrained = True).net
checkpoint = torch.load(checkpoint_path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
net.load_state_dict(new_state_dict)

net = net.to(devices[0])
net.eval()  
# ----------------------predict on the test set-------------------
# initialize variables
classes = ['Liquid', 'Powder', 'SingleCrystal'] 
true_labels = []
predicted_labels = []
wrong_predictions = []  # save predicted incorrect images and their information 
class_correct = np.zeros(len(classes))
class_total = np.zeros(len(classes))

# predict and collect true/predicted labels 

for X, labels in test_iter:
    # print(f"test_iter:{test_iter}")
    with torch.no_grad():
        y_hat = net(X.to(devices[0]))
        labels = labels.to(devices[0])

    _, predicted = torch.max(y_hat, 1)
    true_labels.extend(labels.cpu().numpy())
    predicted_labels.extend(predicted.cpu().numpy())
    # images = X.cpu().numpy()


    # calculate correct predictions and total numbers for each class
    for i in range(len(labels)):
        # print(f"labels[{i}]:{labels[i]}")
        # print(f"y_hat[i]: {y_hat[i]}")
        class_correct[labels[i]] += 1 if predicted[i] == labels[i] else 0
        class_total[labels[i]] += 1

        if predicted[i] != labels[i]:
            true_label = classes[labels[i]]
            predicted_label = classes[predicted[i]]
            img = X[i].cpu().numpy().transpose((1, 2, 0))
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            # convert numpy array to PIL Image
            img = Image.fromarray(img)
            # adjust the brightness and contrast
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(3)  # Increase brightness (1.0 is the original brightness)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Increase contrast (1.0 is the original contrast)
            wrong_predictions.append((img, true_label, predicted_label))
       

# calculate accuracy for each class
class_accuracy = class_correct / class_total

# print accuracy for each class
for i, cls in enumerate(classes):
    print(f'Accuracy of {cls}: {class_accuracy[i]:.3f}')

# calculate overall accuracy
overall_accuracy = sum(class_correct) / sum(class_total)
print(f'Test Accuracy (Overall): {overall_accuracy:.3f}')

# calculate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(save_dir,
                        f"ConfusionMatrix_batchsize_{train.batch_size}_epoch_{train.num_epochs}_lp_{train.lr_period}_ld_{train.lr_decay}.jpg"))
plt.show()

# output wrong predictionss
for i, (img, true_label, predicted_label) in enumerate(wrong_predictions):
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f'True: {true_label}, Predicted: {predicted_label}')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f'WrongPrediction_{i}.png'))
    plt.show()
    print(f'Image {i}: True Label: {true_label}, Predicted Label: {predicted_label}')


    