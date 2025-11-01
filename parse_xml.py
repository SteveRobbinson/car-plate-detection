import numpy as np
import xml.etree.ElementTree as ET 
import os

myrootdir = '/home/lukas/Desktop/car-plate-dataset/annotations'

def parse_data(rootdir):

	all_annotations = {}

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

			all_annotations[root.find('filename').text] = coords

		return all_annotations

labels = parse_data(myrootdir)