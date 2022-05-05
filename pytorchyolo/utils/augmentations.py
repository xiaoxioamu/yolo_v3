import imgaug.augmenters as iaa
from torchvision import transforms
from utils.transforms import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug


class DefaultAug(ImgAug):
	def __init__(self):
		self.augmentations = iaa.Sequential([
			iaa.Sharpen((0.0, 0.1)),
			iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
			iaa.AddToBrightness((-60, 40)),
			iaa.Fliplr(0.5),
		])


class StrongAug(ImgAug):
	def __init__(self, ):
		self.augmentations = iaa.Sequential([
			iaa.Dropout([0.0, 0.01]),
			iaa.Sharpen((0.0, 0.1)), 
			iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)), 
			iaa.AddToBrightness((-60, 40)), 
			iaa.AddToHue((-20, 20)),
			iaa.Fliplr(0.5),
		])


AUGMENTATION_TRANSFORMS = transforms.Compose([ 
	AbsoluteLabels(), 
	StrongAug(), 
	PadSquare(), 
	RelativeLabels(), 
	ToTensor()
])

if __name__ == "__main__":
	import PIL.Image as Image 
	import numpy as np 

	img_file = 'data/coco/images/train2014/COCO_train2014_000000000009.jpg'
	label_file = 'data/coco/labels/train2014/COCO_train2014_000000000009.txt'
	
	img = np.array(Image.open(img_file))
	boxes = np.loadtxt(label_file).reshape(-1, 5)
	img, boxes = AUGMENTATION_TRANSFORMS((img, boxes))
	img 
