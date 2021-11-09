# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import tensorflow as tf 
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    out=tf.image.resize_nearest_neighbor(conv,[Y,X])
    return out

def conv2dsigmoid(input,W,B):
    return tf.nn.sigmoid(tf.nn.conv2d(input,W,strides=[1,1,1,1],padding='SAME')+B)

def conv2dstride(input,W,B):
    return tf.nn.relu(tf.nn.conv2d(input,W,strides=[1,2,2,1],padding='SAME')+B)
def fc(input,W,B):
    return tf.nn.relu(tf.matmul(input,W)+B)
class colornet():
    def __init__(self,dico,y=224,x=224):
        self.input=tf.placeholder(tf.float32,[None,y,x,1])
        self.inputresized=tf.placeholder(tf.float32,[None,224,224,1])
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
        W=dico[res]['conv0']['W']
        B=dico[res]['conv0']['B']
        fusion=conv2d(fusion,W,B)
       
        W=dico[res]['conv1']['W']
        B=dico[res]['conv1']['B']
        fusion=conv2dzoom(fusion,W,B,y//4,x//4)
        
       
        W=dico[res]['conv2']['W']
        B=dico[res]['conv2']['B']
        fusion=conv2d(fusion,W,B)
       
        W=dico[res]['conv3']['W']
        B=dico[res]['conv3']['B']
        fusion=conv2dzoom(fusion,W,B,y//2,x//2)
       
        W=dico[res]['conv4']['W']
        B=dico[res]['conv4']['B']
        fusion=conv2d(fusion,W,B)
       
        W=dico[res]['conv5']['W']
        B=dico[res]['conv5']['B']
        fusion=conv2dsigmoid(fusion,W,B)
        self.output=fusion # fini

        
        

#%%
coloriseur=colornet(dico,224,224)
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

image=np.zeros((1,224,224,1))
image[0,20:40,20:40,0]=0.5
image[0,100:140,100:140,0]=0.3
out=sess.run(coloriseur.output,feed_dict={coloriseur.input:image,coloriseur.inputresized:image})
sess.close()
#%%
plt.imshow(out.reshape((224,224,2))[:,:,1])



