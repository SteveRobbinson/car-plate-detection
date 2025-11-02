import numpy as np
import xml.etree.ElementTree as ET 
import os
from PIL import Image
from torchvision.transforms import v2
from torchvision.io import decode_image
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

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
			# all_annotations[root.find('filename').text] = coords
			all_annotations = pd.DataFrame(data)

		return all_annotations

annotations = parse_data(annotationsdir)
annotations = annotations.explode('coordinates')



transforms = v2.Compose([
	v2.Resize((640, 640)),
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
		img = Image.open(img_path).convert('RGB')
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


train_features, train_labels = next(iter(train_loader))

img = train_features[0].squeeze()
label = train_labels[0]

plt.imshow(img, cmap='gray')
plt.show()
print(label)


