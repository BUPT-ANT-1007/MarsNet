## MarsNet
MarsNet combines Transformers and U-Net to detect rocks on Mars for Zhurong Rover. On one hand, the Transformers encodes global context which lacks in the convolution neural network (CNN) features. On the other hand, the detailed high-resolution spatial information from CNN features enables precise localization. Hybrid dilated convolution (HDC) is also added into MarsNet to enlarge the receptive fields of the network. Larger receptive fields assist to detect huge rocks.
### Index
1. [Environment]
2. [Dataset]
3. [How2train]
4. [How2predict]
5. [Reference]


### Environment
torch==1.10.1    
torchvision==0.11.2



### Dataset
the Dataset we use is TWMARS which can be find at https://github.com/BUPT-ANT-1007/Mars-surface-image-segmentation-dataset-for-Tianwen-1-Mission.git

### How2train 

1.This paper uses VOC format for training.
2.Before training, put the label files in the SegmentationClass under VOC2007 folder under VOCdevkit folder.    
3.Before training, put the picture files in JPEGImages under VOC2007 folder under VOCdevkit folder.    
4.Before training, use voc_ annotation. py to generate the corresponding TXT.    
5.Run train.py to start training.  


### How2predict

1. Follow the training steps.    
2. In unet.py, modify model_path, backbone, and num_class make it correspond to the trained files.    **model_ path corresponds to the weight file under the logs folder**.

```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_voc.pth',
    #--------------------------------#
    #   所需要区分的类的个数+1
    #--------------------------------#
    "num_classes"   : 3,
    #--------------------------------#
    #   所使用的的主干网络：vgg、resnet50   
    #--------------------------------#
    "backbone"      : "vgg",
    #--------------------------------#
    #   输入图片的大小
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"         : True,
    #--------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------#
    "cuda"          : True,
}
```
3. Run predict.py  
```python
img/street.jpg
```      

## Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bubbliiiing/unet-pytorch
