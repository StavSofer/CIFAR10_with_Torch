# CIFAR10_with_Torch

**Goals**:</br>
  *Parameters < 50,000</br>
  *Accuracy > 80%
  
 **Results** :+1:</br>
 Accuracy = 0.826</br>
 Number of parameters = 49706</br>
 Number of training epochs = 300

### Model - 
```
  (1): nn.BatchFlip
  (2): cudnn.SpatialConvolution(3 -> 32, 5x5, 1,1, 2,2)
  (3): cudnn.SpatialBatchNormalization
  (4): nn.ReLU
  (5): cudnn.SpatialConvolution(32 -> 32, 1x1)
  (6): cudnn.SpatialBatchNormalization
  (7): nn.ReLU
  (8): cudnn.SpatialConvolution(32 -> 32, 1x1)
  (9): cudnn.SpatialBatchNormalization
  (10): nn.ReLU
  (11): cudnn.SpatialConvolution(32 -> 32, 1x1)
  (12): cudnn.SpatialBatchNormalization
  (13): nn.ReLU
  (14): cudnn.SpatialMaxPooling(2x2, 2,2)
  (15): cudnn.SpatialConvolution(32 -> 64, 3x3, 1,1, 1,1)
  (16): cudnn.SpatialBatchNormalization
  (17): nn.ReLU
  (18): cudnn.SpatialConvolution(64 -> 64, 1x1)
  (19): cudnn.SpatialBatchNormalization
  (20): nn.ReLU
  (21): cudnn.SpatialMaxPooling(2x2, 2,2)
  (22): nn.Dropout(0.450000)
  (23): cudnn.SpatialConvolution(64 -> 32, 3x3, 1,1, 1,1)
  (24): cudnn.SpatialBatchNormalization
  (25): nn.ReLU
  (26): cudnn.SpatialConvolution(32 -> 32, 1x1)
  (27): cudnn.SpatialBatchNormalization
  (28): nn.ReLU
  (29): cudnn.SpatialAveragePooling(5x5, 2,2)
  (30): nn.View(128)
  (31): nn.Linear(128 -> 10)
  (32): nn.LogSoftMax
```
