# ColNet - Colorize Black and White Images

The main objective of this project was to colorize black and white images. To do this, we used an existing neural network called ColNet and tried to analyze it and better the results. In the end, we also focused on colorizing "NIR" (near-infrared) images with a new network derived from ColNet.

![image](https://user-images.githubusercontent.com/72748466/199738014-f1bd980a-2d67-48de-aa1b-db2629f0bdc2.png)



## The Original Network ColNet
‍

ColNet will deduce, from a binary image, a chrominance image in two canals (a and b from the lab color space), which will thus give the colored image by converting from lab to RGB.

To compute the chrominance image, the NN contains several "branches" which will each scan the image with different scales. This method permits the "detection" of each object while at the same time "understand" whether the picture is inside or outside for instance.

In our project, we did not have total access to the original project and thus had to recreate the network from the paper and retrain it. Our lack of computer power forced us to expect disappointing results, or at least not as good as their results.

![image](https://user-images.githubusercontent.com/72748466/199738442-0aad1379-c845-43ef-bc26-8e6c79327ecb.png)


## Our Results and Improvements
‍

After a quite simple training on vegetals classes, the results are already quite satisfactory. Here, we can compare the original images with the results.

Because the training classes are mainly green pictures, we notice that the network adds too much green sometimes but we've seen (see the report for more information) that training on more diverse classes bring way better results for these problems.

The main issue we noticed on our results was a lack of saturated colors, as the loss was defined in a way that the error would be higher if the colors were saturated. Thus, we modified the loss to favor highly saturated colors, but the results were not really convincing.

![image](https://user-images.githubusercontent.com/72748466/199738576-e8952205-54d7-4950-ba39-754f4d0aa40b.png)


## Colorizing NIR Images
‍

After working on "classic" black and white images, we decided to focus on another idea which was colorizing NIR images. While it may not seem that different at first, as NIR images are also technically black and white images, it is in fact a whole new level of difficulty. Indeed, using our network on NIR images didn't give the expected results because NIR images are clearly different of simple B&W images.

After some experimentations on various modifications of the NN (which you can read about in our report), we decided to add another output branche that would deduce from the NIR images the luminance canal (the B&W image). Thus, rather than only computing the color of the image, we would ask the network to compute all the image. While this network was obviously far more complex, it was also very promising as it could, if working, be used in way more situations than just NIR images as long as we had another re-training (or just a fine tuning). It thus combined the complex system of multi-scaling features of ColNet and a more complete output than ColNet.

Obviously, the training was way longer and, because of our limited equipment and limited time, we couldn't do the complete training we wanted. Indeed, when we stopped the training, the loss was still decreasing and no clear sign of overfitting appeared. However, the results (can be seen below) are really promising, as the colors and the forms are well-respected. The main issue is clearly the blur, which was reducing through the training and which may have been partially removed with a better training. We can also identify a few errors but they are very few ones among the pictures. Finally, we notice a lack of saturation which was also on ColNet and we could use some of the previously mentionned methods to solve this problem.

![image](https://user-images.githubusercontent.com/72748466/199738779-a08d0d78-92b7-4594-b571-7977d4ad36f7.png)


**Please check the PDF report for more information.**
