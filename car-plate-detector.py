from lxml import etree
from PIL import Image
from torchvision.transforms import v2
import torch
from pathlib import Path
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.ops import generalized_box_iou_loss

annotationsdir = Path('/home/lukas/Desktop/car-plate-dataset/annotations')
imagedir = Path('/home/lukas/Desktop/car-plate-dataset/images')


def parse_data(rootdir):
    
    query = '//bndbox'
    data = []
    broken_data = []

    for child in rootdir.iterdir():
        tree = etree.parse(child)
        bndbox_list = tree.xpath(query)

        for elements in bndbox_list:
            
            coordinates = elements.xpath('*/text()')
            
            if len(coordinates) < 4:
                broken_data.append(child.name)
                continue

            data.append({
                'file_name': child.stem + '.png',
                'coordinates' : [int(x) for x in coordinates]
                })

    data = pd.DataFrame(data)

    return data

annotations = parse_data(annotationsdir)


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
		img_path = self.img_dir / sample['file_name']
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
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 4)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.sigmoid(self.fc(x))

        return x


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(device)
model = CarPlateDetector().to(device)
optimizer = torch.optim.Adam(model.parameters())


def train_one_epoch(epoch_index):
    running_loss = 0

    for i, data in enumerate(train_loader):
        input, label_data = data

        label = torch.stack(label_data)
        optimizer.zero_grad()

        y_hat = model(input)

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

    model.train(True)
    avg_loss = train_one_epoch(i)
    
    running_vloss = 0
    model.eval()

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






