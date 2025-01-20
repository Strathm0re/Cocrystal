import os
import shutil
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.models as models
from d2l import torch as d2l
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import read_dataset 
import model.model as model
# -------------------------set parameters--------------------------
file_dir = r"/home/dell/Research/Disk/1_dataset/Classification_3/upload/"
data_dir = file_dir + r"/dataset/"
save_dir = file_dir + r"/Powder_SingleCrystal_Liquid_ResNet18/save_pth/train_save/"
devices = d2l.try_all_gpus()
num_epochs, lr, wd  = 150, 1e-4, 5e-4
lr_period, lr_decay = 1, 0.99
batch_size = 50
# ---------------------define training function---------------------

def evaluate_loss_gpu(net, data_iter, device, loss_fn):
    net.eval()  # set the model to evaluation mode
    loss_sum, count = 0.0, 0
    with torch.no_grad():  # disable gradient calculation
        for features, labels in data_iter:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            l = loss_fn(outputs, labels)
            loss_sum += l.sum().item()
            count += features.size(0)
    return loss_sum / count  # calculate average loss

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    # # view parameters in the optimizer
    # for param_group in trainer.param_groups:
    #     print(param_group['params'])
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    train_loss_values = []
    valid_loss_values = []
    train_acc_values = []
    valid_acc_values = []
    epochs = []

    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)

        print(" ")
        
        with tqdm(total=len(train_iter), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as t:
            for i, (features, labels) in enumerate(train_iter):
                timer.start()
                l, acc = d2l.train_batch_ch13(net, features, labels,
                                              loss, trainer, devices)
                metric.add(l, acc, labels.shape[0])
                current_lr = scheduler.get_last_lr()[0]  # get current learning rate
                timer.stop()

                t.set_postfix(loss=f"{metric[0] / metric[2]:.3f}", acc=f"{metric[1] / metric[2]:.3f}",lr=f"{current_lr}")
                t.update()
            
        scheduler.step()

        if valid_iter is not None:
            valid_loss = evaluate_loss_gpu(net, valid_iter, devices[0], loss)
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            valid_loss_values.append(valid_loss)
            valid_acc_values.append(valid_acc)


        train_loss_values.append(metric[0] / metric[2])
        train_acc_values.append(metric[1] / metric[2])
        epochs.append(epoch + 1)

        # plot: Training and Validation Metrics
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss_values, label='Train Loss')
        plt.plot(epochs, train_acc_values, label='Train Accuracy')
        if valid_iter is not None:
            valid_epochs = range(num_epochs)
            plt.plot(epochs, valid_loss_values, label='Valid Loss')
            plt.plot(epochs, valid_acc_values, label='Valid Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.title('Training and Validation Metrics')
        plt.legend()
        plt.grid(True)
        if epoch != num_epochs-1:
            plt.pause(0.1)  
            plt.close()  
        if epoch == num_epochs-1:
            plt.savefig(os.path.join(save_dir,
                                     f"train_batchsize_{batch_size}_epoch_{num_epochs}_lp_{lr_period}_ld_{lr_decay}.jpg"))

    
    # calculate average loss
    avg_train_loss = np.mean(train_loss_values)
    avg_train_acc = np.mean(train_acc_values)
    measures = f'Average train loss: {avg_train_loss:.3f}, Average train acc: {avg_train_acc:.3f}'
    if valid_iter is not None:
        avg_valid_loss = np.mean(valid_loss_values)
        avg_valid_acc = np.mean(valid_acc_values)
        measures += f', Average valid loss: {avg_valid_loss:.3f}, Average valid acc: {avg_valid_acc:.3f}'
    print(measures)

    # print final training and validation accuracy
    # measures = (f'train loss {metric[0] / metric[2]:.3f}, '
    #             f'train acc {metric[1] / metric[2]:.3f}')
    # if valid_iter is not None:
    #     measures += f', valid loss {valid_loss:.3f}, valid acc {valid_acc:.3f}'
    # print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
    #       f' examples/sec on {str(devices)}')
    torch.save(net.state_dict(), os.path.join(save_dir,
                                              f'train_resnet18_batchsize_{batch_size}_epoch_{num_epochs}_lp_{lr_period}_ld_{lr_decay}.pth'))

if __name__ == '__main__':
    # ----------------read dataset (include augmentation)--------------
    loader = read_dataset.reader(data_dir, batch_size)
    train_ds, valid_ds, train_valid_ds = loader.train_ds, loader.valid_ds, loader.train_valid_ds
    train_iter, valid_iter, train_valid_iter = loader.train_iter, loader.valid_iter, loader.train_valid_iter

    # ------------------------------load model----------------------------
    net = model.get_net(pretrained = True).net
    loss = nn.CrossEntropyLoss(reduction="none")
    # --------------------------train the model---------------------------
    # Start training   
    # train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)
    # print(f"batch_size, num_epochs, lr, lr_period, lr_decay = {batch_size}, {num_epochs}, {lr}, {lr_period}, {lr_decay}")

    # -------retrain model with training and validation set------------
    train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)
    print(f"retrain: batch_size, num_epochs, lr, lr_period, lr_decay = {batch_size}, {num_epochs}, {lr}, {lr_period}, {lr_decay}")





