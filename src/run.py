from src import loadImages, ConvNet
import json
import os
import cv2
import sys
import numpy as np
from skimage import segmentation, color, filters, io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import csv

net = ConvNet()
net.load_state_dict(torch.load('model.pt'))
net.eval()
json_content = list()
list1 =[]
list2 = []

arg = sys.argv[1:]

v = "-v" in arg

if len(arg) == 0:
    print('No arguments detected. Demo with visualization for 1 image.')
    image_files = ['SA_20220620-102621_8ka1kmwpywxv_incision_crop_0.jpg']
    
    output_file = 'output.csv'
    v = False
else:
    if v:
        image_files = arg[2:]
        print('Starting detection with visualization.')
    else:
        image_files = arg[1:]
        print('Starting detection without visualization.')
    output_file = arg[0]
    
class CustomTransform:
    def __call__(self, image):
        # Příklad vlastní transformace: rozostření obrázku
        image = transforms.functional.adjust_contrast(image, 1.4)

        #image = (filters.roberts(color.rgb2gray(image)))
        #image = (morphology.erosion(image, morphology.rectangle(3,3)))
        image = color.rgb2gray(image)
        ###thr = image.max()/3
        ##image[image <= thr] = 0
        #image[image > thr] = 1
        image = color.gray2rgb(image)
        #image = np.expand_dims(image, 0)
        #image = np.einsum('ijk->kij',image)
        return image
        
for filename in image_files:
    print(filename)
    img = cv2.imread(filename)
    image = Image.fromarray(img)
    # Transformace pro předzpracování obrázků
    transform = transforms.Compose([
        transforms.RandomAdjustSharpness(p = 1.0, sharpness_factor = 2.0),
        CustomTransform(),
        transforms.ToPILImage(),
        transforms.Resize((60, 150)),  # Změna velikosti obrázku
        ##transforms.RandomHorizontalFlip(p = 0.5),
        #transforms.RandomVerticalFlip(p = 0.5), # Náhodné  převrácení
        #transforms.RandomAdjustSharpness(1,1.3),
        #transforms.ColorJitter(brightness=0.1), # Náhodná změna jasu
        transforms.ToTensor(), # Převod obrázku na tensor
        #transforms.Normalize(mean=[0.385, 0.356, 0.306], std=[0.129, 0.124, 0.125]) ,
        ])
    image = transform(image).unsqueeze(0)
    
    # Ensure model is in eval mode
    net.to('cpu')
    net.eval()
    
    # Choose the layer you want to visualize
    # Here, we assume your model has a convolutional layer named 'conv1'
    layer = net.conv1
    layer2 = net.conv2
    layer3 = net.conv3
    layer4 = net.conv4
    
    # Register hook
    activation = {}
    activation2 = {}
    activation3 = {}
    activation4 = {}
    def get_activation(name):
        def hook(net, input, output):
            activation[name] = output.detach()
        return hook
    def get_activation2(name):
        def hook2(net, input, output):
            activation2[name] = output.detach()
        return hook2
    def get_activation3(name):
        def hook3(net, input, output):
            activation3[name] = output.detach()
        return hook3
    def get_activation4(name):
        def hook4(net, input, output):
            activation4[name] = output.detach()
        return hook4

    handle = layer.register_forward_hook(get_activation('conv1'))
    handle2 = layer2.register_forward_hook(get_activation2('conv2'))
    handle3 = layer3.register_forward_hook(get_activation3('conv3'))
    handle4 = layer4.register_forward_hook(get_activation4('conv4'))
    
    outputs = net(image)
    res = outputs.detach().numpy()
    print("Počet stehů: " + str(np.argmax(res)))
    list1.append(filename)
    list2.append("Počet stehů: " + str(np.argmax(res)))
    
    handle.remove()
    handle2.remove()
    handle3.remove()
    handle4.remove()

    feature_map = activation['conv1'].squeeze()
    feature_map2 = activation2['conv2'].squeeze()
    feature_map3 = activation3['conv3'].squeeze()
    feature_map4 = activation4['conv4'].squeeze()
    if v:
        # Plot feature maps
        fig, axarr = plt.subplots(min(feature_map.shape[0]+2, 6), figsize=(8, 16))
        
        image = np.einsum('ijk->jki',image.squeeze(0)
                        .numpy())
        axarr[0].imshow(img)
        axarr[1].imshow(image)
    
        axarr[2].imshow(feature_map[0])
    
        axarr[3].imshow(feature_map2[0])
        axarr[4].imshow(feature_map3[0])
    
        axarr[5].imshow(feature_map4[0])

    
        plt.show()

with open(output_file, 'w', newline='') as csvfile: 
    writer = csv.writer(csvfile) 
    writer.writerow(['Filename', 'Amount of stitch'])  # Writing the header 
    for i in range(len(list1)): 
        writer.writerow([list1[i], list2[i]])  # Writing the data from both lists to the CSV file 

