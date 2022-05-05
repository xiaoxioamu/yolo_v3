from torch.utils.data import Dataset 
import torch.nn.functional as F 
import torch
import random 
import os
import glob 
import warnings
import numpy as np 
from PIL import Image


def resize(image, size):
	image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
	return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):

	def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
		with open(list_path, 'r') as file:
			self.img_files = file.readlines()

		self.label_file = []
		for path in self.img_files:
			image_dir = os.path.dirname(path)
			label_dir = "labels".join(image_dir.rsplit("images", 1))
			assert label_dir != image_dir, \
				f"Image path must contain a folder named 'images' \n{image_dir}"
			label_file = os.path.join(label_dir, os.path.basename(path))
			label_file = os.path.splitext(label_file)[0] + '.txt'
			self.label_file.append(label_file)

		self.img_size = img_size
		self.max_object = 100
		self.multiscale = multiscale 
		self.min_size = self.img_size - 3 * 32
		self.max_size = self.img_size + 3 * 32
		self.batch_count = 0
		self.transform = transform 


	def __getitem__(self, index):

		# -------------
		# Image
		# -------------
		try:
			img_path = self.img_files[index % len(self.img_files)].rstrip()	
			img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
	
		except Exception:
			print(f"Could not read image '{img_path}'.")
			return


		# ------------
		# Label
		# ------------
		try:
			label_path = self.label_file[index % len(self.img_files)].rstrip()

			# Ignore warning if file is empty
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				boxes = np.loadtxt(label_path).reshape(-1, 5)

		except Exception:
			print(f"Could not read label '{label_path}")
			return 

		# -----------
		# Transform
		# -----------

		if self.transform:
			try:
				img, bb_targets = self.transform((img, boxes))
			except Exception:
				print("Could not apply transform")
				return 
		
		return img_path, img, bb_targets 

	def collate_fn(self, batch):
		self.batch_count += 1

		# Drop invalid images
		batch = [data for data in batch if data is not None]

		paths, imgs, bb_targets = list(zip(*batch))

		# Selects new image size every tenth batch
		if self.multiscale and self.batch_count % 10 == 0:
			self.img_size = random.choice(
				range(self.min_size, self.max_size + 1, 32))

		# Resize images to input shape
		imgs = torch.stack([resize(img, self.img_size) for img in imgs])

		# Add sample index to targets
		for i, boxes in enumerate(bb_targets):
			boxes[:, 0] = i
		bb_targets = torch.cat(bb_targets, 0)

		return paths, imgs, bb_targets

	def __len__(self):
		return len(self.img_files)
		



if __name__ == "__main__":
	from torch.utils.data import DataLoader
	from augmentations import AUGMENTATION_TRANSFORMS
	from utils import worker_seed_set

	img_path = "data/custom/train.txt"
	img_size = 416
	multiscale_training = False 
	batch_size= 2
	n_cpu= 2

	dataset = ListDataset(
		img_path,
		img_size=img_size,
		multiscale=multiscale_training,
		transform=AUGMENTATION_TRANSFORMS)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=n_cpu,
		pin_memory=True,
		collate_fn=dataset.collate_fn,
		worker_init_fn=worker_seed_set)
	
	iter_batch = iter(dataloader)
	mini_batch = next(iter_batch)
	print(mini_batch)