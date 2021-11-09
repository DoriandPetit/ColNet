import os
import torch
import numpy as np
import torchvision.datasets
import torchvision.transforms
from skimage import color, io
from random import randint
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class HandleGrayscale(object):
    """Feeds the pipeline with 3 - channel image.
    
    All transformations below work with RGB images only.
    If a 1-channel grayscale image is given, it's converted to
    equivalent 3-channel RGB image.
    """
    def __call__(self, image):
        if len(image.shape) < 3:
            image = color.gray2rgb(image)
        return image


class RandomCrop(object):
    """Randomly crops an image to size x size."""
    
    def __init__(self, size=224):
        self.size = size
        
    def __call__(self, image,resize=False):


        if resize :
            cropped = cv2.resize(image,(self.size,self.size),interpolation = cv2.INTER_AREA)
            off_h,off_w =0,0
        else :    
            h, w, _ = image.shape
            assert min(h, w) >= self.size

            off_h = randint(0, h - self.size)
            off_w = randint(0, w - self.size)

            cropped = image[off_h:off_h+self.size, off_w:off_w+self.size]
        
        #print(image.shape,cropped.shape)
        assert cropped.shape == (self.size, self.size, 3)
        return cropped, off_h,off_w

    
class Rgb2LabNorm(object):
    """Converts an RGB image to normalized image in LAB color sapce.
    
    Both (L, ab) channels are in [0, 1].
    """
    def __call__(self, image):
        assert image.shape == (224, 224, 3)
        img_lab = color.rgb2lab(image)
        img_lab[:,:,:1] = img_lab[:,:,:1] / 100.0
        img_lab[:,:,1:] = (img_lab[:,:,1:] + 128.0) / 256.0
        return img_lab
    
    
class ToTensor(object):
    """Converts an image to torch.Tensor.
    
    Note:
        Images are Height x Width x Channels format. 
        One needs to convert them to CxHxW.
    """
    def __call__(self, image):
        
        assert image.shape == (224, 224, 3)
        
        transposed =  np.transpose(image, (2, 0, 1)).astype(np.float32)
        image_tensor = torch.from_numpy(transposed)

        assert image_tensor.shape == (3, 224, 224)
        return image_tensor

    
class SplitLab(object):
    """Splits tensor LAB image to L and ab channels."""
    def __call__(self, image):
        assert image.shape == (3, 224, 224)
        L  = image[:1,:,:]
        ab = image[1:,:,:]
        return (L, ab)
    
    

def modif_dataset(root,subfolder) :
    """ Function made to modify the NIR EPFL dataset into the good architecture (as described in the report).
    """
    
    path = os.path.join(root,subfolder)
    size = len(os.listdir(path))//2
    rgb,nir=0,0
    filenames = os.listdir(path)

    filenames = np.sort(filenames)
    for k in range(size) :
        new_path = os.path.join(path,str(k))
        os.makedirs(new_path)
    for i in range(2*size) :
        print(filenames[i],i)
    
        if "_rgb.tiff" in filenames[i]:
            os.rename(os.path.join(path,filenames[i]), os.path.join(path,str(i//2),filenames[i]))
            rgb = 1
        if "_nir.tiff" in filenames[i]:
            os.rename(os.path.join(path,filenames[i]), os.path.join(path,str(i//2),filenames[i]))
            nir = 1
        if ((rgb ==1) and (nir==1)) :
            rgb,nir=0,0
            continue

#modif_dataset("../data/places10/train","indoor")        
#modif_dataset("../data/places10/train","mountain")
#modif_dataset("../data/places10/train","oldbuilding")
#modif_dataset("../data/places10/train","street")        
#modif_dataset("../data/places10/train","urban")
#modif_dataset("../data/places10/train","water")

from glob import glob


class ImagesDateset:
    """Custom dataset for loading and pre-processing images. Made for the NIR EPFL dataset, with the modified architecture, each __get_item__ call analyzes one subfolder."""

    def __init__(self, root):
        """Initializes the dataset and loads images. 



        Args:
            root: a directory from which images are loaded
        """
        
        self.root = root

        self.composed = torchvision.transforms.Compose(
            [HandleGrayscale(), Rgb2LabNorm(),ToTensor()]
        )
        print(os.walk(root))
        
        self.labels = [x[1] for x in os.walk(self.root)][0]
        print("####################")
        print(self.labels)
        print("####################")
        
        self.len=[]
        for k in range(len(self.labels)):
            self.len.append(len([name for name in os.listdir(os.path.join(self.root,self.labels[k]))]))
     

    def __len__(self):
        return (len(glob(os.path.join(self.root,"*/*/"))))


        
    def __getitem__(self, idx):
        """Gets a subfolder, and returns L of the nir image, Lab of the RGB image and the label
    
        """  
        
        subdirs = self.labels
        labels=[]
        
        for k in range(len(subdirs)) :
            labels += [subdirs[k]] *len(os.listdir(os.path.join(self.root,subdirs[k])))
        
        size = [0]
        for k in self.len :
            size.append(k+size[-1])
        if idx == 0:
            new_idx = 0

        elif idx in size :
            size.append(idx)
            size.sort()
            index = size.index(idx)+1
            size.remove(idx)
            new_idx = idx - size[index-1]
            
        else:
            size.append(idx)
            size.sort()
            index = size.index(idx)
            size.remove(idx)
            new_idx = idx - size[index-1]
        
        path = os.path.join(self.root,labels[idx],str(new_idx))

        filenames = os.listdir(path)
        filenames = np.sort(filenames)
        
        image_rgb = io.imread(os.path.join(path,filenames[1]))
        image_nir = io.imread(os.path.join(path,filenames[0]))
        
        crop = RandomCrop(size=224)
        image_rgb, off_h,off_w = crop(image_rgb,resize=True)
        Lab = self.composed(image_rgb)

        L = self.composed(cv2.resize(image_nir,(224,224),interpolation = cv2.INTER_AREA))[0]

        
        return L,Lab,labels[idx]

    
    def get_name(self, idx):
        path = os.path.normpath(self.imgs[idx][0])
        name = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path))
        return label + "-" + name