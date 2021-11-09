

from skimage import io as skio
from skimage.color import rgb2lab, lab2rgb
import glob
import numpy as np
import matplotlib.pyplot as plt

import os


def rgb_to_various_grey(source_directory,target_directory):
    """
    takes RGB images from a source directory, 
    save a 1 chanel image which's the average of the 3 chanel into target directory
    save a 1 chanel image which's r chanel into target directory
    save a 1 chanel image which's g chanel into target directory
    save a 1 chanel image wicho's b chanel into target directory
    
    Parameters
    ----------
    source_directory : directory containing images which we want to uncolor
    target_directory : directory where the images with 1 chanel will be saved

    Returns
    -------
    None.

    """
    

    for filepath in glob.glob(source_directory+'/*.jpg'):
        
        
        filename = os.path.basename(filepath)

        im_rgb=skio.imread(filepath)
        
        im_r = im_rgb[:,:,0]
        im_g = im_rgb[:,:,1]
        im_b = im_rgb[:,:,2]
        im_mean = np.uint8((im_r/3 + im_g/3 + im_b/3))
        im_weigted_mean = np.uint8((0.2125*im_r + 0.7154*im_g + 0.0721*im_b))
        
        skio.imsave(target_directory+'/r_'+filename, im_r)
        skio.imsave(target_directory+'/g_'+filename, im_g)
        skio.imsave(target_directory+'/b_'+filename, im_b)
        skio.imsave(target_directory+'/mean_'+filename, im_mean)
        skio.imsave(target_directory+'/weigted_mean_'+filename, im_weigted_mean)
 
        
  

    