# **An attempt at Image Super Resolution: Part I**

Haven't you been amused at watching TV series when the detective zooms into the image of a crime scene and he finds a clue that leads to the culprit? Sounds so sci-fi! Well this is my attempt to code that out. While Super Resolution has been there for quite some time now , being used for various tasks. One of them being delivering high resolution content such as 1080 resolution movies through streaming while consuming lesser bandwidth. Surprising!? I know!. 



## **What is Super Resolution?**

As the name very much suggests  , super resolution is basically to create higher resolution of images given lower resolution images. While this is not very straight as it sounds. You can't just open an 512 X 512 image in an editor and resize into a 1024 X 1024 and call it as a higher resolution image . Rather you would see blurred pixels . 

Lets code that out and see for ourselves

`lr_img = cv2.imread('Data/0797_lr.png')`

`print(lr_img.shape)
plt.imshow(lr_img)`



```python
(320, 320, 3)
```

![Low Resolution Image](https://github.com/anuda/Medium/blob/master/SR_1/0797_lr.png)

Now lets resize the image to 2X. Below is the code and the output of the image.  If you observe , you can see the image getting blurred. This is because just doubling images just can't create and fill in the missing information. 

`hr_img = cv2.resize(lr_img,(lr_img.shape[1]*2,lr_img.shape[0]*2),interpolation = cv2.INTER_NEAREST)`
`print(hr_img.shape)`
`plt.imshow(hr_img)`

```python
(640, 640, 3)
```

![2X image](https://github.com/anuda/Medium/blob/master/SR_1/hr_img_nn_art.png)

Image source: https://data.vision.ee.ethz.ch/cvl/DIV2K/

*Interpolation* above does the trick here in filling in the missing values. A quick highlight what each of it means:

| Enumerator                              |                                |
| :-------------------------------------- | ------------------------------ |
| INTER_NEAREST Python: cv2.INTER_NEAREST | nearest neighbor interpolation |
| INTER_LINEAR Python: cv2.INTER_LINEAR   | bilinear interpolation         |
| INTER_CUBIC Python: cv2.INTER_CUBIC     | bicubic interpolation          |

The above ones are used for zooming purposes. 

Source: https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

## Why do we need super resolution at all?

Super Resolution bases its applications in the field of computer vision. Computer vision has wide spread applications in medical domains, automated machines which use pattern recognition or object detection at its core. Low quality or noisy images can affect the quality of the model inference .   

##### If data can't be created, is Super Resolution achievable?

With the progress of research in the field of Deep Learning , multiple researchers have been able to propose and build architectures that are able to achieve such feats. This is possible due the reason that Neural networks can fill in the gaps when they are trained over pairs of low resolution and their corresponding higher resolution images.

![nn_sr](https://github.com/anuda/Medium/blob/master/SR_1/nn_sr.png)



While there are multiple such architectures and designs that work on similar lines we will try to build one super resolution pipeline using Resnet-18 based model. We will together build one in the next article.

Few papers to inspire all of us:

- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network : https://arxiv.org/abs/1609.04802
- Image Super-Resolution Using Deep Convolutional Networks: https://arxiv.org/abs/1501.00092

Icon Credits:

<div>Icons made by <a href="https://www.flaticon.com/authors/smashicons" title="Smashicons">Smashicons</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>

## 

