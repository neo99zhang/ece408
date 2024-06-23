# Baseline Statistic

## Layer
```
Conv-GPU==
B: 1000, M: 4, C: 1, H: 86, W: 86, K: 7
Conv-GPU==
B: 1000, M: 16, C: 4, H: 40, W: 40, K: 7
```

## 10000
[link](http://s3.amazonaws.com/files.rai-project.com/userdata/build-5fb38c415fb7931872910b09.tar.gz)
```
Test Accuracy: 0.8714
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 649.011626 ms
Layer 1 OpTime: 649.031402 ms
Layer 1 LayerTime: 1217.220132 ms
Layer 2 GPUTime: 1847.408382 ms
Layer 2 OpTime: 1847.426174 ms
Layer 2 LayerTime: 2260.892457 ms
```

## 1000
```
Test Accuracy: 0.886
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 72.010842 ms
Layer 1 OpTime: 72.026042 ms
Layer 1 LayerTime: 133.616172 ms
Layer 2 GPUTime: 203.609995 ms
Layer 2 OpTime: 203.626187 ms
Layer 2 LayerTime: 249.46679 ms
```

## Shared Memory
```
Test Accuracy: 0.886
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 2.369066 ms
Layer 1 OpTime: 2.388458 ms
Layer 1 LayerTime: 60.460769 ms
Layer 2 GPUTime: 14.130401 ms
Layer 2 OpTime: 14.148417 ms
Layer 2 LayerTime: 57.182514 ms
```

## Const
```
Test Accuracy: 0.886
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 71.544442 ms
Layer 1 OpTime: 71.557722 ms
Layer 1 LayerTime: 134.707699 ms
Layer 2 GPUTime: 195.359905 ms
Layer 2 OpTime: 195.380193 ms
Layer 2 LayerTime: 238.588381 ms
```



## TILE_WIDTH 32 -> 20


### 1000
```
Test Accuracy: 0.886
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 1.81221 ms
Layer 1 OpTime: 1.831186 ms
Layer 1 LayerTime: 63.198314 ms
Layer 2 GPUTime: 5.970545 ms
Layer 2 OpTime: 5.986544 ms
Layer 2 LayerTime: 50.443454 ms
```

## unroll loop
[Link](http://s3.amazonaws.com/files.rai-project.com/userdata/build-5fb3d3c65fb79336707f616f.tar.gz)
### 10000
```
Test Accuracy: 0.8714
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 16.170759 ms
Layer 1 OpTime: 16.189351 ms
Layer 1 LayerTime: 597.435326 ms
Layer 2 GPUTime: 52.334772 ms
Layer 2 OpTime: 52.357716 ms
Layer 2 LayerTime: 478.444808 ms
```

### 1000
```
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 1.62124 ms
Layer 1 OpTime: 1.643 ms
Layer 1 LayerTime: 61.578199 ms
Layer 2 GPUTime: 5.193093 ms
Layer 2 OpTime: 5.209125 ms
Layer 2 LayerTime: 48.923996 ms
```