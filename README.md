### This is a PyTorch implementation of the paper: [Abnormal cervical cells detection using YOLOV7 with  spatial and channel attention decoupled head]
![ScreenShot](/images/framework.jpg)

## Requirements

* Pytorch 1.12.0
* Torchvision 0.13.0
* At least 1 GPU

## Datasets

* [dataset](https://github.com/kuku-sichuan/ComparisonDetector)

## Training
Run the train.py file for network training
 "--cfg" select the configuration file for the network model
"--data" choose your own dataset
"--epochs" train epoch
"--batch-size" total batch for all GPUs
"--name" save to project/name 
## Testing
Run test.py for network testing
"--weights" configure the weights of the trained network

## Citation
It will be provided soon:

```
@ARTICLE{
}
```
## References 

* [Relation Networks](https://github.com/WongKinYiu/yolov7)




