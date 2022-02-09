import torch
import torch.nn as nn
import os
import requests
import os.path


from torch.nn import Module
#%%

from collections import OrderedDict
from torch import nn
from vonenet.modules import VOneBlock
from vonenet.back_ends import ResNetBackEnd, Bottleneck, AlexNetBackEnd, CORnetSBackEnd
from vonenet.params import generate_gabor_param
import numpy as np


def VOneNet(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
            model_arch='resnet50', image_size=224, visual_degrees=8, ksize=25, stride=4,skipS_n_C=False):


    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

    gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}


    # Conversions
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                           simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size)

    if model_arch:
        bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

        if model_arch.lower() == 'resnet50':
            print('Model: ', 'VOneResnet50')
            model_back_end = ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3])
        elif model_arch.lower() == 'alexnet':
            print('Model: ', 'VOneAlexNet')
            model_back_end = AlexNetBackEnd()
        elif model_arch.lower() == 'cornets':
            print('Model: ', 'VOneCORnet-S')
            model_back_end = CORnetSBackEnd()

        model = nn.Sequential(OrderedDict([
            ('vone_block', vone_block),
            ('bottleneck', bottleneck),
            ('model', model_back_end),
        ]))
    else:
        print('Model: ', 'VOneNet')
        model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.gabor_params = gabor_params
    model.arch_params = arch_params

    return model

#%%
FILE_WEIGHTS = {'alexnet': 'vonealexnet_e70.pth.tar', 'resnet50': 'voneresnet50_e70.pth.tar',
                'resnet50_at': 'voneresnet50_at_e96.pth.tar', 'cornets': 'vonecornets_e70.pth.tar',
                'resnet50_ns': 'voneresnet50_ns_e70.pth.tar'}


class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def get_model(model_arch='resnet50', pretrained=True, map_location='cpu', **kwargs):
    """
    Returns a VOneNet model.
    Select pretrained=True for returning one of the 3 pretrained models.
    model_arch: string with identifier to choose the architecture of the back-end (resnet50, cornets, alexnet)
    """

    if pretrained and model_arch:
        url = f'https://vonenet-models.s3.us-east-2.amazonaws.com/{FILE_WEIGHTS[model_arch.lower()]}'
        #home_dir = os.environ['HOME']
        home_dir  = os.path.expanduser('~')
        vonenet_dir = os.path.join(home_dir, '.vonenet')
        weightsdir_path = os.path.join(vonenet_dir, FILE_WEIGHTS[model_arch.lower()])
        if not os.path.exists(vonenet_dir):
            os.makedirs(vonenet_dir)
        if not os.path.exists(weightsdir_path):
            print('Downloading model weights to ', weightsdir_path)
            r = requests.get(url, allow_redirects=True)
            open(weightsdir_path, 'wb').write(r.content)

        ckpt_data = torch.load(weightsdir_path, map_location=map_location)

        stride = ckpt_data['flags']['stride']
        simple_channels = ckpt_data['flags']['simple_channels']
        complex_channels = ckpt_data['flags']['complex_channels']
        k_exc = ckpt_data['flags']['k_exc']

        noise_mode = ckpt_data['flags']['noise_mode']
        noise_scale = ckpt_data['flags']['noise_scale']
        noise_level = ckpt_data['flags']['noise_level']

        model_id = ckpt_data['flags']['arch'].replace('_','').lower()

        model = globals()[f'VOneNet'](model_arch=model_id, stride=stride, k_exc=k_exc,
                                      simple_channels=simple_channels, complex_channels=complex_channels,
                                      noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                                     )

        if model_arch.lower() == 'resnet50_at':
            ckpt_data['state_dict'].pop('vone_block.div_u.weight')
            ckpt_data['state_dict'].pop('vone_block.div_t.weight')
            model.load_state_dict(ckpt_data['state_dict'])
        else:
            model = Wrapper(model)
            model.load_state_dict(ckpt_data['state_dict'])
            model = model.module

        model = nn.DataParallel(model)
    else:
        model = globals()[f'VOneNet'](model_arch=model_arch, **kwargs)
        model = nn.DataParallel(model)

    model.to(map_location)
    return model


#%%
from PIL import Image
import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
import numpy as np
import pandas
import tqdm
import fire

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
def test1(modelname,data_path, output_path, isVOne,isResnet,layer='decoder', sublayer='avgpool', numlayer=None,layername =None, time_step=0, imsize=224):

    model = get_model(model_arch=modelname, pretrained=True)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().numpy()
        _model_feats.append(output)

    try:
        m = model.module
    except:
        m = model
    if isVOne:
        print('VOneBlock')
        model_layer = getattr(getattr(m, layer), sublayer)
    else:
        if isResnet:
            print('Backbone is ResNet')
            model_block = getattr(getattr(m, layer), sublayer)
            model_layer = getattr(getattr(model_block, numlayer), layername)
        else:
            print('Backbone is Alexnet or Cornet')
            model_block = getattr(getattr(m, layer), sublayer)
            model_layer = getattr(model_block, numlayer)

    model_layer.register_forward_hook(_store_feats)

    model_feats = []
    model_pic_name = []
    with torch.no_grad():
        model_feats = []
        fnames = sorted(glob.glob(os.path.join(data_path, '*.*')))
        if len(fnames) == 0:
            raise FileNotFoundError(f'No files found in {data_path}')
        for fname in tqdm.tqdm(fnames):
            print(fname)
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise FileNotFoundError(f'Unable to load {fname}')
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            _model_feats = []
            model(im)
            #print(np.shape(_model_feats))
            model_feats.append(_model_feats[time_step])
            #model_pic_name.append(fname)
        model_feats = np.concatenate(model_feats)
        #model_pic_name = np.concatenate(model_pic_name)


    if output_path is not None:
        fname = f'VOneNet-{modelname}_{layer}_{sublayer}_{numlayer}_{layername}_{time_step}.npy'

        np.save(os.path.join(output_path, fname), model_feats)
    #return model_pic_name
#%%
input_path = r'D:\OneDrive - Washington University in St. Louis\GitHub\CORnet\heatmap_pictures'
output_path = r'D:\OneDrive - Washington University in St. Louis\GitHub\CORnet\output_images'



model_names = ['alexnet','resnet50_ns','resnet50_at','resnet50','cornets'] #['alexnet',
for themodel in model_names:
    print('======using ' + themodel)
    #v1_model = get_model(model_arch=themodel, pretrained=True,skipS_n_C=False)
    if themodel == 'alexnet':
        print('in alexnet')
        layers_to_test = {
                          'vone_block': 'output',
                          'model': 'features_0', # conv2
                          'model': 'features_1', # relu2
                          'model': 'features_3', # conv3
                          'model': 'features_4', # relu3
                          'model': 'features_5', # conv4
                          'model': 'features_6', # relu4
                          'model': 'features_7', # conv5
                          'model': 'features_8', # relu5
                          }
        for key, value in layers_to_test.items():

            layer = key
            sublayer = value
            if layer == 'vone_block':
                test1(themodel, input_path, output_path, isVOne=True, isResnet=False,
                      layer=layer,
                      sublayer=sublayer, imsize=224)
            else :

                sublayers = ['features_0','features_1',  'features_3', 'features_4', 'features_5', 'features_6', 'features_7','features_8']
                for t_sublayer in sublayers:
                    print("  in: " + t_sublayer)
                    sublayer = t_sublayer[:-2]
                    numlayer = t_sublayer[-1]
                    # if t_sublayer == 'features_0':
                    #     sublayer = 'features'
                    #     numlayer = '0'   # 1the ReLu after conv2
                    # elif t_sublayer == 'features_3':
                    #     sublayer = 'features'
                    #     numlayer = '3'   #4 the ReLu after conv3
                    # elif t_sublayer == 'features_5':
                    #     sublayer = 'features'
                    #     numlayer = '5'  #6 the ReLu after conv4
                    # else:    # 'model': 'features_7'
                    #     sublayer = 'features'
                    #     numlayer = '7' #8 the ReLu after conv4

                    test1(themodel, input_path, output_path, isVOne=False, isResnet=False,
                              layer=layer,
                              sublayer=sublayer,
                              numlayer=numlayer,
                              imsize=224)



    elif themodel == 'resnet50_ns'  or themodel == 'resnet50_at' or themodel == 'resnet50':

        print('in Resnet')
        layers_to_test = {
                          'vone_block': 'output',
                          'model': 'layer1',   #block 2
                          'model': 'layer2',   #block 3
                          'model': 'layer3',   #block 4
                          'model': 'layer4'    #block 5
                          }
        for key, value in layers_to_test.items():
            layer = key
            sublayer = value
            if layer == 'vone_block':
                test1(themodel, input_path, output_path, isVOne=True, isResnet=False,
                      layer=layer,
                      sublayer=sublayer, imsize=224)
            else:
                sublayers = ['layer1', 'layer2', 'layer3', 'layer4']
                for sublayer in sublayers:
                    layernames = ['conv3','relu']
                    for layername in layernames:
                        if sublayer == 'layer1':
                            numlayer = '2'
                            #layername = 'conv3'  # 2 -> conv3 -> bn3 -> relu
                        elif sublayer == 'layer2':
                            numlayer = '3'
                            #layername = 'conv3'  # 3 -> conv3 -> bn3 -> relu
                        elif sublayer == 'layer3':
                            numlayer = '5'
                            #layername = 'conv3'  # 5 -> conv3 -> bn3 -> relu
                        else:  # sublayer == 'layer4':
                            numlayer = '2'
                            #layername = 'conv3'  # 2 -> conv3 -> bn3 -> relu
                        test1(themodel, input_path, output_path, isVOne=False, isResnet=True,
                              layer=layer,
                              sublayer=sublayer,
                              numlayer=numlayer,
                              layername=layername, imsize=224)

    elif themodel == 'cornets':
        print('in Cornet')
        layers_to_test = {
                         'vone_block': 'output',
                          'model': 'V2',
                          'model': 'V4',
                          'model': 'IT',
                          }
        for key, value in layers_to_test.items():
            layer = key
            sublayer = value
            if layer == 'vone_block':


                test1(themodel, input_path, output_path, isVOne=True, isResnet=False,
                      layer=layer,
                      sublayer=sublayer, imsize=224)

            else:
                sublayers = ['V2','V4', 'IT']
                for sublayer in sublayers:
                    numlayers = ['conv3','nonlin3']
                    for numlayer in numlayers:
                        if sublayer == 'V2': #repeated twice

                            time_step = 1
                        elif sublayer == 'V4': #repeated 4 times
                            time_step = 3
                        else:
                            time_step = 1 #repeated twice
                        #numlayer = 'conv3'  # conv3 -> norm_3_(time_step) -> nonlin3
                        test1(themodel, input_path, output_path, isVOne=False, isResnet=False,
                              layer=layer,
                              sublayer=sublayer,
                              numlayer=numlayer,
                              time_step=time_step, imsize=224)

#%%
themodel = 'alexnet'
v1_model = get_model(model_arch=themodel, pretrained=True)
from layer_hook_utils import layername_dict, register_hook_by_module_names, get_module_names, named_apply
model_names_list = get_module_names(v1_model, [3, 715,715], device="cpu" )


