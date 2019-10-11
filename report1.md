# Report

Names:
- Jimmy He (miaoh2)
- Steven Chan (chchan2)
- 

Team name: `gutfeeling`

## Part 1: Include a list of all kernels that collectively consume more than 90% of the program time.

```
Time(%)      Time     Calls       Avg       Min       Max  Name
32.35%  35.909ms        20  1.7955ms  1.0880us  33.591ms  [CUDA memcpy HtoD]
17.91%  19.883ms         1  19.883ms  19.883ms  19.883ms  volta_scudnn_128x64_relu_interior_nn_v1
17.24%  19.143ms         4  4.7858ms  4.7804ms  4.7903ms  volta_gcgemm_64x32_nt
8.54%  9.4788ms         4  2.3697ms  1.9489ms  3.1196ms  void fft2d_c2r_32x32<float, bool=0, bool=0, unsigned int=0, bool=0, bool=0>(float*, float2 const *, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)
7.17%  7.9621ms         1  7.9621ms  7.9621ms  7.9621ms  volta_sgemm_128x128_tn
6.54%  7.2647ms         2  3.6324ms  25.600us  7.2391ms  void op_generic_tensor_kernel<int=2, float, float, float, int=256, cudnnGenericOp_t=7, cudnnNanPropagation_t=0, cudnnDimOrder_t=0, int=1>(cudnnTensorStruct, float*, cudnnTensorStruct, float const *, cudnnTensorStruct, float const *, float, float, float, float, dimArray, reducedDivisorArray)
5.75%  6.3869ms         4  1.5967ms  1.2541ms  2.0263ms  void fft2d_r2c_32x32<float, bool=0, unsigned int=0, bool=0>(float2*, float const *, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)
```

`memcpy` is not a CUDA kernel.

Most time-consuming CUDA kernels are the following:
- `volta_scudnn_128x64_relu_interior_nn_v1`
- `volta_gcgemm_64x32_nt`
- `fft2d_c2r_32x32`
- `volta_sgemm_128x128_tn`
- `op_generic_tensor_kernel`
- `fft2d_r2c_32x32`


## Part 2: Include a list of all CUDA API calls that collectively consume more than 90% of the program time.


```
40.50%  2.93490s        22  133.40ms  13.586us  1.55540s  cudaStreamCreateWithFlags
33.51%  2.42821s        24  101.18ms  58.414us  2.42308s  cudaMemGetInfo
21.13%  1.53108s        19  80.583ms  1.1530us  409.69ms  cudaFree
```

Most time-consuming API calls:
- `cudaStreamCreateWithFlags`
- `cudaMemGetInfo`
- `cudaFree`


## Part 3: Include an explanation of the difference between kernels and API calls

It is worth noting the total wall time elapsed executing the API call far supercedes the amount of time spent by the kernels. This means that API calls are essentially mandatory overhead to the computation kernels.

In this case, kernels are program sections, mostly made up of CUDA code, to be allocated into SMs for executing a arithmetic comptation, e.g. convolution. These make the most use of GPUs' arithematic and logical units (ALUs). API calls are program mostly initiated from the host side to manage GPU functions, for example managing GPU memory, allocating memory blocks and controlling SMs to launch a kernel. These calls make the most use of GPUs' control units.

## Part 4: Show output of rai running MXNet on the CPU; List program run time

```
✱ Running /usr/bin/time python m1.1.py
Loading fashion-mnist data... done
Loading model... done
New Inference
EvalMetric: {'accuracy': 0.8154}
17.03user 4.80system 0:08.98elapsed 243%CPU (0avgtext+0avgdata 6045928maxresident)k
0inputs+2824outputs (0major+1601833minor)pagefaults 0swaps
```

Program run time: 8.98 seconds wall time; 17.03 seconds User CPU and 4.80 seconds System CPU.

## Part 5: Show output of rai running MXNet on the CPU; List program run time

```
✱ Running /usr/bin/time python m1.2.py
Loading fashion-mnist data... done
Loading model... done
New Inference
EvalMetric: {'accuracy': 0.8154}
4.88user 3.39system 0:04.62elapsed 179%CPU (0avgtext+0avgdata 2953204maxresident)k
0inputs+
1712outputs (0major+731724minor)pagefaults 0swaps
```

Program run time: 4.62 seconds wall time; 4.88 seconds User CPU and 3.39 seconds System CPU. Note that the CPU time metrics may not reflect the actually runtime as GPU usage might have registered as `iowait` for CPU.

## Part 6: CPU Implementation; List whole program execution time; List op times


