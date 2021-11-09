# -*- coding: utf-8 -*-
"""
Script Fourni et Modifié permettant de recoder le réseau Colnet avec les poids pickle fournis. Projet IMA206
"""
from random import randint
import cv2
import tensorflow as tf 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

nomfichierpickle='colornet.pickle'
#%%

f=open(nomfichierpickle,'rb')
dico=pickle.load(f)
f.close()
#%%
def conv2d(input,W,B):
    return tf.nn.relu(tf.nn.conv2d(input,W,strides=[1,1,1,1],padding='SAME')+B)

def conv2dzoom(input,W,B,Y,X):
    conv=tf.nn.relu(tf.nn.conv2d(input,W,strides=[1,1,1,1],padding='SAME')+B)
    out=tf.compat.v1.image.resize_nearest_neighbor(conv,[Y,X])
    return out

def conv2dsigmoid(input,W,B):
    return tf.nn.sigmoid(tf.nn.conv2d(input,W,strides=[1,1,1,1],padding='SAME')+B)

def conv2dstride(input,W,B):
    return tf.nn.relu(tf.nn.conv2d(input,W,strides=[1,2,2,1],padding='SAME')+B)
def fc(input,W,B):
    return tf.nn.relu(tf.matmul(input,W)+B)
class colornet():
    def __init__(self,dico,y=224,x=224):
        tf.compat.v1.disable_eager_execution() ### NOT TO BE KEPT
        self.input=tf.compat.v1.placeholder(tf.float32,[None,y,x,1])
        self.inputresized=tf.compat.v1.placeholder(tf.float32,[None,y,x,1])
        res='LowLevelFeatures' # debut du reseau
        print(res)
        localnet=self.input
        globalnet=self.inputresized
        
        W=dico[res]['conv0']['W']
        B=dico[res]['conv0']['B']
        globalnet=conv2dstride(globalnet,W,B)
        localnet=conv2dstride(localnet,W,B)
       
        W=dico[res]['conv1']['W']
        B=dico[res]['conv1']['B']
        globalnet=conv2d(globalnet,W,B)
        localnet=conv2d(localnet,W,B)
       
        W=dico[res]['conv2']['W']
        B=dico[res]['conv2']['B']
        globalnet=conv2dstride(globalnet,W,B)
        localnet=conv2dstride(localnet,W,B)
       
        W=dico[res]['conv3']['W']
        B=dico[res]['conv3']['B']
        globalnet=conv2d(globalnet,W,B)
        localnet=conv2d(localnet,W,B)
       
        W=dico[res]['conv4']['W']
        B=dico[res]['conv4']['B']
        globalnet=conv2dstride(globalnet,W,B)
        localnet=conv2dstride(localnet,W,B)
       
        W=dico[res]['conv5']['W']
        B=dico[res]['conv5']['B']
        globalnet=conv2d(globalnet,W,B)
        localnet=conv2d(localnet,W,B)
       
        # partie global features
        res='GlobalFeaturespartial'
        print(res)
        W=dico[res]['conv0']['W']
        B=dico[res]['conv0']['B']
        globalnet=conv2dstride(globalnet,W,B)
       
        W=dico[res]['conv1']['W']
        B=dico[res]['conv1']['B']
        globalnet=conv2d(globalnet,W,B)
        
        W=dico[res]['conv2']['W']
        B=dico[res]['conv2']['B']
        globalnet=conv2dstride(globalnet,W,B)
       
        W=dico[res]['conv3']['W']
        B=dico[res]['conv3']['B']
        globalnet=conv2d(globalnet,W,B)
        
        globalnet=tf.reshape(globalnet,[-1,7*7*512])
        
        W=dico[res]['fc0']['W']
        B=dico[res]['fc0']['B']
        globalnet=fc(globalnet,W,B)
        
        W=dico[res]['fc1']['W']
        B=dico[res]['fc1']['B']
        globalnet=fc(globalnet,W,B)
        
        res='GlobalFeaturesfinal'
        print(res)
        W=dico[res]['fc0']['W']
        B=dico[res]['fc0']['B']
        globalnet=fc(globalnet,W,B) #la partie globale est un vecteur 256
        
        # retour au localnet avant fusion
        res='MidLevelFeatures'
        print(res)
        W=dico[res]['conv0']['W']
        B=dico[res]['conv0']['B']
        localnet=conv2d(localnet,W,B)
       
        W=dico[res]['conv1']['W']
        B=dico[res]['conv1']['B']
        localnet=conv2d(localnet,W,B)
        print('debut fusion')
        # FUSION 
        globalnet=tf.expand_dims(tf.expand_dims(globalnet,1),1)
        globalnet=tf.tile(globalnet,[1,y//8,x//8,1])
        
        fusion=tf.concat([localnet,globalnet],axis=-1)
        
        # colorisation
        res='Colorization'
        print(res)
        #print(fusion.shape)
        W=dico[res]['conv0']['W']
        B=dico[res]['conv0']['B']
        fusion=conv2d(fusion,W,B)
        #print(fusion.shape)
        
        W=dico[res]['conv1']['W']
        B=dico[res]['conv1']['B']
        fusion=conv2dzoom(fusion,W,B,y//4,x//4)
        #print(fusion.shape)
       
        W=dico[res]['conv2']['W']
        B=dico[res]['conv2']['B']
        fusion=conv2d(fusion,W,B)
        #print(fusion.shape)
        
        W=dico[res]['conv3']['W']
        B=dico[res]['conv3']['B']
        fusion=conv2dzoom(fusion,W,B,y//2,x//2)
        #print(fusion.shape)
        
        W=dico[res]['conv4']['W']
        B=dico[res]['conv4']['B']
        fusion=conv2d(fusion,W,B)
        #print(fusion.shape)
       
        W=dico[res]['conv5']['W']
        B=dico[res]['conv5']['B']
        fusion=conv2dsigmoid(fusion,W,B)
        #print(fusion.shape)
        fusion=tf.compat.v1.image.resize_nearest_neighbor(fusion,[y,x])
        self.output=fusion # fini

        
        

#%%
coloriseur=colornet(dico,224,224)
sess=tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

#%%


file = "ansel_colorado_1941.png"

image = cv2.imread("./"+file)
image = cv2.resize(image, dsize=(224,224), interpolation=cv2.INTER_CUBIC)

#%%
if image.shape != (224,224,3):
    image = random_resize(image,224)
img_lab = color.rgb2gray(image)
assert image.shape == (224, 224, 3)

        
image = img_lab.reshape((1,224,224,1))
print("Img Shape : ",image.shape)
out=sess.run(coloriseur.output,feed_dict={coloriseur.input:image,coloriseur.inputresized:image})
# print(out.shape)
sess.close()
#%%
out = out.reshape((224,224,2)).astype(np.float64)
image = image.reshape((224,224,1)).astype(np.float64)


# Convert to numpy and unnnormalize
L = image *100
ab_out = out * 254.0 - 127.0
    
# Stack layers  
img_stack = np.dstack((L, ab_out))
   
img_stack = img_stack.astype(np.float64)	
out = color.lab2rgb(img_stack)

plt.figure()
plt.imshow(out)
plt.show()

plt.figure()
plt.imshow(cv2.imread("./"+file))
plt.show()



