import torch

import torch.nn as nn

from collections import namedtuple


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    
class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False),
                                                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False),
                                       nn.PReLU(depth),
                                       nn.Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False),
                                       nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut
    
class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                                                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
                                       nn.PReLU(depth),nn.Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
                                       nn.BatchNorm2d(depth),
                                       SEModule(depth,16))
        
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, stride = 2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class Backbone(nn.Module):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                            nn.BatchNorm2d(64),
                                            nn.PReLU(64))
        if input_size == 112:
            self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                              nn.Dropout(drop_ratio),
                                              nn.Flatten(),
                                              nn.Linear(512 * 7 * 7, 512),
                                              nn.BatchNorm1d(512, affine=affine))
        else:
            self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                              nn.Dropout(drop_ratio),
                                              Flatten(),
                                              nn.Linear(512 * 14 * 14, 512),
                                              nn.BatchNorm1d(512, affine=affine))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)
    
def IR_101(input_size):
    model = Backbone(input_size, 100, mode="ir", drop_ratio=0.4, affine=False)
    return model

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se') # modified resnet50
        self.facenet.load_state_dict(torch.load('model_ir_se50.pth'))
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
    def extract_feats(self, x):
        x = torch.nn.functional.interpolate(x, (256, 256), mode='bilinear')
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats
    
    def forward(self, y_hat, y):
        n_samples = y_hat.size(0)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count