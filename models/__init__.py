import torch
from models import simnet, alexnet, vgg, resnet, \
    senet, resnext, densenet, simplenetv1, \
    efficientnetv2, googlenet, xception, mobilenetv2, \
    inceptionv3, wideresnet, shufflenetv2, squeezenet, mnasnet


def load_model(model_name, in_channels=3, num_classes=10):
    print('-' * 50)
    print('LOAD MODEL:', model_name)
    print('-' * 50)

    model = None
    if model_name == 'simnet':
        model = simnet.simnet()
    elif model_name == 'alexnet':
        model = alexnet.alexnet(in_channels, num_classes)
    elif model_name == 'vgg16':
        model = vgg.vgg16_bn(in_channels, num_classes)
    elif model_name == 'resnet34':
        model = resnet.resnet34(in_channels, num_classes)
    elif model_name == 'resnet50':
        model = resnet.resnet50(in_channels, num_classes)
    elif model_name == 'senet34':
        model = senet.seresnet34(in_channels, num_classes)
    elif model_name == 'wideresnet28':
        model = wideresnet.wide_resnet28_10(in_channels, num_classes)
    elif model_name == 'resnext50':
        model = resnext.resnext50(in_channels, num_classes)
    elif model_name == 'densenet121':
        model = densenet.densenet121(in_channels, num_classes)
    elif model_name == 'simplenetv1':
        model = simplenetv1.simplenet(in_channels, num_classes)
    elif model_name == 'efficientnetv2s':
        model = efficientnetv2.effnetv2_s(in_channels, num_classes)
    elif model_name == 'efficientnetv2l':
        model = efficientnetv2.effnetv2_l(in_channels, num_classes)
    elif model_name == 'googlenet':
        model = googlenet.googlenet(in_channels, num_classes)
    elif model_name == 'xception':
        model = xception.xception(in_channels, num_classes)
    elif model_name == 'mobilenetv2':
        model = mobilenetv2.mobilenetv2(in_channels, num_classes)
    elif model_name == 'inceptionv3':
        model = inceptionv3.inceptionv3(in_channels, num_classes)
    elif model_name == 'shufflenetv2':
        model = shufflenetv2.shufflenetv2(in_channels, num_classes)
    elif model_name == 'squeezenet':
        model = squeezenet.squeezenet(in_channels, num_classes)
    elif model_name == 'mnasnet':
        model = mnasnet.mnasnet(in_channels, num_classes)
    return model


def load_modules(model, model_layers=None):
    assert model_layers is None or type(model_layers) is list

    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            modules.append(module)

    modules.reverse()  # reverse order
    if model_layers is None:
        model_modules = modules
    else:
        model_modules = []
        for layer in model_layers:
            model_modules.append(modules[layer])

    print('-' * 50)
    print('Model Layers:', model_layers)
    print('Model Modules:', model_modules)
    print('Model Modules Length:', len(model_modules))
    print('-' * 50)

    return model_modules
