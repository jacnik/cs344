# Compiling and running snippets:
```sh
# compile
nvcc reduce.cu -o reduce.out

# run
./reduce.out 0 # 0 = reduce with global memory; 1 = reduce with shared memory
```


## Reduce algorithm
[Nvidia article about parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

```
array d_in:
 ___ ___ ___ ___ ___ ___ ___ ___ ___ ____ ____ ____ ____ ____ ____ ____
| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
 ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾
size = 16
threads = 4
blocks = size / threads = 4

Block 0:            Block 1:           Block 2:             Block 3:
 ___ ___ ___ ___    ___ ___ ___ ___    ___ ____ ____ ____    ____ ____ ____ ____
| 1 | 2 | 3 | 4 |  | 5 | 6 | 7 | 8 |  | 9 | 10 | 11 | 12 |  | 13 | 14 | 15 | 16 |
 ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾    ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾    ‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾    ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾

Example kernel for Block 1:

blockDim.x = 4
blockIdx.x = 1

d_in index:    4   5   6   7
              ___ ___ ___ ___
             | 5 | 6 | 7 | 8 |
              ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾
tid:           0   1   2   3
myId:          4   5   6   7



sdata [shared array, separate for each block, of same size as number of threads]

after copying from d_id to sdata { sdata[tid] = d_in[myId]; }

array sdata:
              ___ ___ ___ ___
             | 5 | 6 | 7 | 8 |
              ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾
sdata index:   0   1   2   3


loop:

    s = blockDim.x / 2 = 2

    first iteration:
if (tid < s) -> runs only for treads 0 and 1

 ___ ___ ___ ___     ___ ___ ___ ___
| 5 | 6 | 7 | 8 |   | 5 | 6 | 7 | 8 |
 ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾     ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾
  ^      /                ^      /
  |     /                 |     /
  |    /                  |    /
  +----                   +----

sdata after first iteration:
 ____ ____ ___ ___
| 12 | 14 | 7 | 8 |
 ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾

s >>= 1 -> 2 >> 1 = 1

    second iteration:
if (tid < s) -> runs only for treads 0

sdata after second iteration:
 ____ ____ ___ ___
| 26 | 14 | 7 | 8 |
 ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾

:end loop

thread 0 writes result (sdata[0]) for this block back to d_out array at blockIdx.x (1)

d_out after all 5 blocks run:
 ____ ____ ____ ____
| 10 | 26 | 42 | 58 |
 ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾
After that this array is reduced again by one block, the final sum is at index 0
 ```
