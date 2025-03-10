from collections import OrderedDict
import torch
import torchvision
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
def get_layer_names(model):
    layername = []
    conv_cnt = 0
    fc_cnt = 0
    pool_cnt = 0
    do_cnt = 0
    for layer in list(model.features)+list(model.classifier):
        if isinstance(layer, nn.Conv2d):
            conv_cnt += 1
            layername.append("conv%d" % conv_cnt)
        elif isinstance(layer, nn.ReLU):
            name = layername[-1] + "_relu"
            layername.append(name)
        elif isinstance(layer, nn.MaxPool2d):
            pool_cnt += 1
            layername.append("pool%d"%pool_cnt)
        elif isinstance(layer, nn.Linear):
            fc_cnt += 1
            layername.append("fc%d" % fc_cnt)
        elif isinstance(layer, nn.Dropout):
            do_cnt += 1
            layername.append("dropout%d" % do_cnt)
        else:
            layername.append(layer.__repr__())
    return layername

#%
# Readable names for classic CNN layers in torchvision model implementation.
layername_dict ={"alexnet":["conv1", "conv1_relu", "pool1",
                            "conv2", "conv2_relu", "pool2",
                            "conv3", "conv3_relu",
                            "conv4", "conv4_relu",
                            "conv5", "conv5_relu", "pool3",
                            "dropout1", "fc6", "fc6_relu",
                            "dropout2", "fc7", "fc7_relu",
                            "fc8",],
                "vgg16":['conv1', 'conv1_relu',
                         'conv2', 'conv2_relu', 'pool1',
                         'conv3', 'conv3_relu',
                         'conv4', 'conv4_relu', 'pool2',
                         'conv5', 'conv5_relu',
                         'conv6', 'conv6_relu',
                         'conv7', 'conv7_relu', 'pool3',
                         'conv8', 'conv8_relu',
                         'conv9', 'conv9_relu',
                         'conv10', 'conv10_relu', 'pool4',
                         'conv11', 'conv11_relu',
                         'conv12', 'conv12_relu',
                         'conv13', 'conv13_relu', 'pool5',
                         'fc1', 'fc1_relu', 'dropout1',
                         'fc2', 'fc2_relu', 'dropout2',
                         'fc3'],
                 "densenet121":['conv1',
                                 'bn1',
                                 'bn1_relu',
                                 'pool1',
                                 'denseblock1', 'transition1',
                                 'denseblock2', 'transition2',
                                 'denseblock3', 'transition3',
                                 'denseblock4',
                                 'bn2',
                                 'fc1']}


#%% Hooks based methods to get layer and module names
def named_apply(model, name, func, prefix=None):
    # resemble the apply function but suits the functions here.
    cprefix = "" if prefix is None else prefix + "." + name
    for cname, child in model.named_children():
        named_apply(child, cname, func, cprefix)

    func(model, name, "" if prefix is None else prefix)


def get_module_names(model, input_size, device="cpu", ):
    module_names = OrderedDict()
    module_types = OrderedDict()
    module_spec = OrderedDict()
    def register_hook(module, name, prefix):
        # register forward hook and save the handle to the `hooks` for removal.
        def hook(module, input, output):
            # during forward pass, this hook will append the ReceptiveField information to `receptive_field`
            # if a module is called several times, this hook will append several times as well.
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(module_names)
            module_names[str(module_idx)] = prefix + "." + class_name +name
            module_types[str(module_idx)] = class_name
            module_spec[str(module_idx)] = OrderedDict()
            if isinstance(input[0], torch.Tensor):
                module_spec[str(module_idx)]["inshape"] = tuple(input[0].shape[1:])
            else:
                module_spec[str(module_idx)]["inshape"] = (None,)
            if isinstance(output, torch.Tensor):
                module_spec[str(module_idx)]["outshape"] = tuple(output.shape[1:])
            else:
                module_spec[str(module_idx)]["outshape"] = (None,)
        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                # and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    else:
        x = torch.rand(2, *input_size).type(dtype)

    # create properties
    # receptive_field = OrderedDict()
    module_names["0"] = "Image"
    module_types["0"] = "Input"
    module_spec["0"] = OrderedDict()
    module_spec["0"]["inshape"] = input_size
    module_spec["0"]["outshape"] = input_size
    hooks = []

    # register hook recursively at any module in the hierarchy
    # model.apply(register_hook)
    named_apply(model, "", register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("------------------------------------------------------------------------------")
    line_new = "{:>14}  {:>12}   {:>12}   {:>12}   {:>25} ".format("Layer Id", "inshape", "outshape", "Type", "ReadableStr", )
    print(line_new)
    print("==============================================================================")
    for layer in module_names:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:7} {:8} {:>12} {:>12} {:>15}  {:>25}".format(
            "",
            layer,
            str(module_spec[layer]["inshape"]),
            str(module_spec[layer]["outshape"]),
            module_types[layer],
            module_names[layer],
        )
        print(line_new)
    return module_names, module_types, module_spec

#%%
def register_hook_by_module_names(target_name, target_hook, model, input_size, device="cpu", ):
    module_names = OrderedDict()
    module_types = OrderedDict()
    target_hook_h = []
    def register_hook(module, name, prefix):
        # register forward hook and save the handle to the `hooks` for removal.
        def hook(module, input, output):
            # during forward pass, this hook will append the ReceptiveField information to `receptive_field`
            # if a module is called several times, this hook will append several times as well.
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_name = prefix + "." + class_name + name
            module_idx = len(module_names)
            module_names[str(module_idx)] = module_name
            module_types[str(module_idx)] = class_name
            if module_name == target_name:
                h = module.register_forward_hook(target_hook)
                target_hook_h.append(h)
        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                # and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    else:
        x = torch.rand(2, *input_size).type(dtype)

    # create properties
    module_names["0"] = "Image"
    module_types["0"] = "Input"
    hooks = []

    # register hook recursively at any module in the hierarchy
    named_apply(model, "", register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()
    if not(len(target_hook_h) == 1):
        print("Cannot hook the layer with the name %s\nAvailable names are listed here"%target_name)
        print("------------------------------------------------------------------------------")
        line_new = "{:>14}  {:>12}   {:>15} ".format("Layer Id", "Type", "ReadableStr", )
        print(line_new)
        print("==============================================================================")
        for layer in module_names:
            print("{:7} {:8} {:>12} {:>15}".format("", layer,
                module_types[layer], module_names[layer],))
        raise ValueError("Cannot hook the layer with the name %s\nAvailable names are listed here"%target_name)
    return target_hook_h, module_names, module_types

#%% Utility code to fetch activation
# def get_activation(name, unit=None):
#     if unit is None:
#         def hook(model, input, output):
#             activation[name] = output.detach()
#     else:
#         def hook(model, input, output):
#             if len(output.shape) == 4:
#                 activation[name] = output.detach()[:, unit[0], unit[1], unit[2]]
#             elif len(output.shape) == 2:
#                 activation[name] = output.detach()[:, unit[0]]
#     return hook
#
# def set_unit(model, name, layer, unit=None):
#     idx = layername.index(layer)
#     layers = list(model.features) + list(model.classifier)
#     handle = layers[idx].register_forward_hook(get_activation(name, unit))
#     return handle
#
# layers = list(classifier.features) + list(classifier.classifier)
# layername = get_layer_names(classifier)
# set_unit(classifier, "score", "conv2", unit=(0,10,10))
# classifier = models.vgg16(pretrained=True)
# classifier.eval()
# score_hk = set_unit(classifier, "score", "fc1", unit=(None, 10, 10))
# img = torch.rand((2, 3, 224, 224))
# out = classifier(img)
# print(activation["score"])
# activation["score"].shape