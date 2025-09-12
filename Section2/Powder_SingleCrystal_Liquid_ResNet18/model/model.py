import torch
from torch import nn
# from torchsummary import summary
import torchvision.models as models

class get_net:
    # use custom resnet18 function to load pretrained model
    def __init__(self, pretrained) -> None:
        self.num_classes = 3
        self.net = self.resnet18(self.num_classes, pretrained)

    def resnet18(self, num_classes, pretrained):
        # create ResNet-18 model
        model = models.resnet18(pretrained)

        if pretrained:
            # load pretrained weights
            state_dict = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet18-5c106cde.pth'
            )
            model.load_state_dict(state_dict)
        
        # freeze all layer parameters except the last layer
        # for param in model.parameters():
        #     param.requires_grad = False

        # replace the last fully connected layer
        num_ftrs = model.fc.in_features
        # model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model.fc = nn.Sequential(
            # nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )

        # set parameters of the last layer to require gradient calculation
        # for param in model.fc.parameters():
        #     param.requires_grad = True

        return model

    def check_update(self):
        # view model structure and parameter information
        net = get_net()
        print(net)
        # check the requires_grad attribute of each layer
        for name, param in net.named_parameters():
            print(f'{name}: {param.requires_grad}')
        # view grad
        for name, param in net.named_parameters():
            if param.grad is not None:
                print(name, param.grad.mean())

        # check model structure and the num of parameter
        # summary(net, input_size=(3, 224, 224))    