# CUDA-GRAPHS
CUDA graph programs in C++

Here are some programs which uses cuda graphs to imcrease the performance.
Cuda graphs are used remove the overheds in launching the kernel as it already dose the work required to launch kernel. So, when there is a program which launchs the same kernel multiple times using cuda graphs will help in boosting the speed of program by reducing the kernel launch time. 

there are 2 ways to implement cuda graphs 
* By stream capture (capture graphs)
* Manually creating (Graph API)

These are the improvments in excution time with cuda graphs (Capture graphs) for the specific operation compared to normal excution.


| Operation    | Without CUDA GRAPH (msec)    |  With CUDA GRAPH (msec)      |
| -------------|:----------------------------:|:----------------------------:|
| MatrixMul    |         3742.34              |         3476.35              |
| SumReduction |         2.948                |         2.106                |
| Convolution  |         8.093                |         7.805                |




These are the improvments in excution time with cuda graphs (Cuda Graph API) for the specific operation compared to normal excution.


| Operation    | Without CUDA GRAPH (msec)    |  With CUDA GRAPH (msec)      |
| -------------|:----------------------------:|:----------------------------:|
| MatrixMul    |         0.387                |         0.347                |
| SumReduction |         2.8153               |         2.427                |
| Convolution  |         12.116               |         11.779               |

* This repository contains the cuda code in c++ for  the above cuda graphs. 
* And also the kernal runtime screenshots are in profile folder. 

