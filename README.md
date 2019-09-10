# image captioning with attention model
This code is a pytorch implementation of the paper [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044).

## Prerequisite
python >= 3.6  
pytorch >= 1.0.0  
torchvision >= 0.4  

## How-to
To train model from scratch, run the following command  
```python train.py```  

To test a trained model(saved as ./model.pt) and get visualized examples, run the following command  
```python test.py```

## Examples
The plot of training loss and validation loss is as below  
![plot of training loss and validation loss](https://github.com/ShelffonZhao/image-captioning-with-attention-model/blob/master/visualization_example/loss.png)

An example of captioning from this model is as below
![example images of attention](https://github.com/ShelffonZhao/image-captioning-with-attention-model/blob/master/visualization_example/example1.PNG)