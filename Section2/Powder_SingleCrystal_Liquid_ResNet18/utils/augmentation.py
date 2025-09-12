import torchvision


def agm():
    #-----------------------------image augmentation------------------------------------
    transform_train = torchvision.transforms.Compose([
    
        torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.0),
                                                ratio=(3.0/4.0, 4.0/3.0)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(degrees=(0, 180)),
        # torchvision.transforms.RandomApply(transforms=[torchvision.transforms.RandomRotation(degrees=(0, 60))], p=0.6),
        # randomly change color
        torchvision.transforms.RandomInvert(p=0.5),
        # randomly change brightness, contrast, and saturation
        torchvision.transforms.ColorJitter(brightness=0.4,
                                        contrast=0.4,
                                        saturation=0.4),
        torchvision.transforms.RandomAdjustSharpness(sharpness_factor = 2, p=0.5),
        # add random noise
        torchvision.transforms.ToTensor(),
        # normalize each channel of an image
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    # during testing, normalize the images only
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2023, 0.1994, 0.2010])])
    
    return transform_train, transform_test