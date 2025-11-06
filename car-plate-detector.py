import numpy as np
import xml.etree.ElementTree as ET 
import os
from PIL import Image
from torchvision.transforms import v2
from torchvision.io import decode_image
import torch
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from torchvision.ops import generalized_box_iou_loss

annotationsdir = '/home/lukas/Desktop/car-plate-dataset/annotations'
imagedir = '/home/lukas/Desktop/car-plate-dataset/images'



def parse_data(rootdir):

	data = []

	for dirpath, _, filenames in os.walk(rootdir):

		for file_name in filenames:

			coords = []
			full_path = os.path.join(dirpath, file_name)
			tree = ET.parse(full_path)
			root = tree.getroot()
			all_objects = root.findall('object')

			for object in all_objects:

				bndbox = object.find('bndbox')

				if bndbox is not None:

					xmin = int(bndbox.find('xmin').text)
					ymin = int(bndbox.find('ymin').text)
					xmax = int(bndbox.find('xmax').text)
					ymax = int(bndbox.find('ymax').text)

					coords.append([xmin, ymin, xmax, ymax])

			data.append({
				'filename' : root.find('filename').text,
				'coordinates' : coords
				})

			all_annotations = pd.DataFrame(data)

		return all_annotations


annotations = parse_data(annotationsdir)
annotations = annotations.explode('coordinates')
print(annotations)



transforms = v2.Compose([
	v2.Resize((320, 320)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
    ])



class CustomImageDataset(Dataset):

	def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
		self.image_labels = annotations_file
		self.img_dir = img_dir
		self.transform = transform 
		self.target_transform = target_transform


	def __len__(self):
		return len(self.image_labels)


	def __getitem__(self, idx):
		sample = self.image_labels.iloc[idx]
		img_path = os.path.join(self.img_dir, sample['filename'])
		img = Image.open(img_path).convert('L')
		label = sample['coordinates']


		if self.transform:
			image = self.transform(img)

		if self.target_transform:
			label = self.target_transform(label)

		return image, label



full_dataset = CustomImageDataset(
	annotations_file = annotations,
	img_dir = imagedir,
	transform = transforms
	)


total_size = len(full_dataset)
training_size = int(total_size * 0.7)
validation_size = int(total_size * 0.15)
test_size = total_size - (training_size + validation_size)


training_data, validation_data, test_data = random_split(
	full_dataset,
	[training_size, validation_size, test_size]
	)


train_loader = DataLoader(training_data, batch_size = 32, shuffle = True, num_workers = 16)
validation_loader = DataLoader(validation_data, batch_size = 32, shuffle = False, num_workers = 16)
test_loader = DataLoader(test_data, batch_size = 32, shuffle = False, num_workers = 16)



class CarPlateDetector(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.hidden_layer1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.hidden_layer2 = nn.MaxPool2d(2)
		
        self.hidden_layer3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.hidden_layer4 = nn.MaxPool2d(2)
     	
        self.hidden_layer5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.hidden_layer6 = nn.MaxPool2d(2)
      	
        self.hidden_layer7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.hidden_layer8 = nn.MaxPool2d(2)
        
        self.hidden_layer9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.hidden_layer10 = nn.MaxPool2d(2)
        
        self.hidden_layer11 = nn.Flatten()
        self.hidden_layer12 = nn.Linear(51200, 512)
        
        self,output_layer = nn.Linear(512, 4)


	def forward(self, x):
		x = F.relu(self.hidden_layer1(x))
		x = self.hidden_layer2(x)
	    
        x = F.relu(self.hidden_layer3(x))
		x = self.hidden_layer4(x)
	    
        x = F.relu(self.hidden_layer5(x))
		x = self.hidden_layer6(x)
        
        x = F.relu(self.hidden_layer7(x))
		x = self.hidden_layer8(x)
        
        x = F.relu(self.hidden_layer9(x))
		x = self.hidden_layer10(x)

        x = self.hidden_layer11(x)
        x = F.relu(self.hidden_layer12(x)) 
        
        x = self.output_layer(x)

		return x



test = CarPlateDetector(1).to(device)
optimizer = torch.optim.Adam(test.parameters())


def train_one_epoch(epoch_index):
    running_loss = 0

    for i, data in enumerate(train_loader):
        input, label_data = data

        label = torch.stack(label_data)
        optimizer.zero_grad()

        y_hat = test(input)

        loss = generalized_box_iou_loss(label, y_hat).diag()
        loss = loss.mean()
        loss.backwards()

        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)

    return avg_loss



epochs = 5

for i in range(epochs):
    print(f"Epoch nr.{i + 1}")

    test.train(True)
    avg_loss = train_one_epoch(i)
    
    running_vloss = 0
    test.eval()

    with torch.no_grad():
        for j, data in enumerate(validation_loader):
            vinput, vlabel_data = data
            vlabel = torch.stack(vlabel_data)
            vyhat = model(vinput)
            vloss = generalized_box_iou_loss(vlabel, vyhat).diag()
            vloss = vloss.mean()
            running_vloss += vloss.item()

    avg_vloss = running_vloss / len(validation_loader)

    print(f"Training loss: {avg_loss}, Validation loss: {avg_vloss}")





