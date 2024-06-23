## unroll
```
Test Accuracy: 0.8714
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 190.287254 ms
Layer 1 OpTime: 208.778795 ms
Layer 1 LayerTime: 842.515319 ms
Layer 2 GPUTime: 203.678432 ms
Layer 2 OpTime: 221.871754 ms
Layer 2 LayerTime: 727.116256 ms
✱ The build folder has been uploaded to http://s3.amazonaws.com/files.rai-project.com/userdata/build-5fdad5865fb7931f394495b9.tar.gz. The data will be present for only a short duration of time.
```


## fuse0
```
Test Accuracy: 0.8714
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 109.849355 ms
Layer 1 OpTime: 109.872171 ms
Layer 1 LayerTime: 690.770979 ms
Layer 2 GPUTime: 63.180233 ms
Layer 2 OpTime: 63.213225 ms
Layer 2 LayerTime: 536.841746 ms
✱ The build folder has been uploaded to http://s3.amazonaws.com/files.rai-project.com/userdata/build-5fdade035fb793333a20e934.tar.gz. The data will be present for only a short duration of time.
```

## fuse1
```
Test Accuracy: 0.8714
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 53.61762 ms
Layer 1 OpTime: 53.64466 ms
Layer 1 LayerTime: 647.839454 ms
Layer 2 GPUTime: 32.125304 ms
Layer 2 OpTime: 32.154488 ms
Layer 2 LayerTime: 505.281082 ms
✱ The build folder has been uploaded to http://s3.amazonaws.com/files.rai-project.com/userdata/build-5fdadf555fb793363a3d6037.tar.gz. The data will be present for only a short duration of time.
```


## Size
```
Conv-GPU==
B: 10000, M: 4, C: 1, H: 86, W: 86, K: 7, H_out: 80, W_out: 80, W_unroll: 6400, H_unroll: 49
Conv-GPU==
B: 10000, M: 16, C: 4, H: 40, W: 40, K: 7, H_out: 34, W_out: 34, W_unroll: 1156, H_unroll: 196
```

## Combine
Test Accuracy: 0.8714
--------------------------------
-           TIMINGS
--------------------------------
Layer 1 GPUTime: 14.642071 ms
Layer 1 OpTime: 14.67119 ms
Layer 1 LayerTime: 618.619493 ms
Layer 2 GPUTime: 29.139086 ms
Layer 2 OpTime: 29.16379 ms
Layer 2 LayerTime: 509.736507 ms
✱ The build folder has been uploaded to http://s3.amazonaws.com/files.rai-project.com/userdata/build-5fdb15b65fb79318b921e575.tar.gz. The data will be present for only a short duration of time.