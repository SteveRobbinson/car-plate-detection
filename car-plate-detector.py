import numpy as np
import xml.etree.ElementTree as ET 
import os
from PIL import Image
from torchvision.transforms import v2
from torchvision.io import decode_image
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset


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
print(annotations)



transforms = v2.Compose([
	v2.Resize((640, 640)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
    ])



class CustomImageDataset(Dataset):

	def __init__(self, annotations, img_dir, transform = None, target_transform = None):
		self.image_labels = annotations
		self.img_dir = img_dir
		self.transform = transform 
		self.target_transform = target_transform


	def __len__(self):
		return len(self.image_labels)


	def __getitem__(self, idx):
		sample = self.image_labels.iloc[idx]
		img_path = os.path.join(self.img_dir, sample['filename'])
		image = decode_image(img_path)
		label = sample['coordinates']

		if self.transform:
			image = self.transform(image)

		if self.target_transform:
			label = self.target_transform(label)

		return image, label

