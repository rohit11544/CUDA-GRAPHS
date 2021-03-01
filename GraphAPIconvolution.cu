#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#define GRAPH_LAUNCH_ITERATIONS  300

using std::cout;
using std::endl;

#define MASK_LENGTH 7

__constant__ int mask[MASK_LENGTH];

void initialize_vector(int* v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = rand() % 100;
    }
}

__global__ void convolution_1d(int* array, int* result, int n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int r = MASK_LENGTH / 2;

    int start = tid - r;

    int temp = 0;

    for (int j = 0; j < MASK_LENGTH; j++) {
        if (((start + j) >= 0) && (start + j < n)) {
            temp += array[start + j] * mask[j];
        }
    }

    result[tid] = temp;
}

void verify_result(int* array, int* mask, int* result, int n) {
    int radius = MASK_LENGTH / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < MASK_LENGTH; j++) {
            if ((start + j >= 0) && (start + j < n)) {
                temp += array[start + j] * mask[j];
            }
        }
        if (temp != result[i]) {
            cout << "NOT SUCCESSFUL" << endl;
            exit(0);
        }
    }
}

void cudaGraphAPIconvolution(int* h_array, int* h_mask, int* h_result, int* d_array, int* d_result, int n, int THREADS, int GRID, size_t bytes_n, size_t bytes_m) {

    initialize_vector(h_array, n);
    cudaStream_t streamForGraph;
    cudaGraph_t graph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t memcpyNode, kernelNode, memsetNode;
    double result_h = 0.0;

    cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking);

    cudaKernelNodeParams kernelNodeParams = { 0 };
    cudaMemcpy3DParms memcpyParams = { 0 };
    cudaMemsetParams memsetParams = { 0 };

    //Adding memcpyParams node
    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(h_array, sizeof(int) * n, n, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(d_array, sizeof(int) * n, n, 1);
    memcpyParams.extent = make_cudaExtent(sizeof(int) * n, 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;

    //Adding memsetParams node
    memsetParams.dst = (void*)d_result;
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(int);
    memsetParams.width = n;
    memsetParams.height = 1;

    cudaGraphCreate(&graph, 0);
    cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams);
    cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);
    nodeDependencies.push_back(memsetNode);
    nodeDependencies.push_back(memcpyNode);

    //Adding Kernal node
    void* kernelArgs[3] = { (void*)&d_array, (void*)&d_result, &n };
    kernelNodeParams.func = (void*)convolution_1d;
    kernelNodeParams.gridDim = dim3(GRID, 1, 1);
    kernelNodeParams.blockDim = dim3(THREADS, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = (void**)kernelArgs;
    kernelNodeParams.extra = NULL;

    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    //Adding memcpyParams node
    memset(&memcpyParams, 0, sizeof(memcpyParams));
    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(d_result, sizeof(int) * n, n, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(h_result, sizeof(int) * n, n, 1);
    memcpyParams.extent = make_cudaExtent(sizeof(int) * n, 1, 1);
    memcpyParams.kind = cudaMemcpyDeviceToHost;

    cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams);
    nodeDependencies.clear();
    nodeDependencies.push_back(memcpyNode);

    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nodes, &numNodes);
    cout << "\nNum of nodes in the graph created manually = " << numNodes << endl;

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaGraph_t clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    cudaGraphClone(&clonedGraph, graph);
    cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++)
    {
        cudaGraphLaunch(graphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
        verify_result(h_array, h_mask, h_result, n);
    }
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Verifying Cloned Graph ..." << endl;
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++)
    {
        cudaGraphLaunch(clonedGraphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
        verify_result(h_array, h_mask, h_result, n);
    }
    cout << "Done! Verifyied successfully" << endl;

    cout << "\nTime taken by using CUDA GRAPH in ms : " << milliseconds / GRAPH_LAUNCH_ITERATIONS << endl;
    
    cudaGraphExecDestroy(graphExec);
    cudaGraphExecDestroy(clonedGraphExec);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(clonedGraph);
    cudaStreamDestroy(streamForGraph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

void NormalConvolution(int* h_array, int* h_mask, int* h_result, int* d_array, int* d_result, const int n, int THREADS, int GRID, size_t bytes_n) {

    cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);

    convolution_1d << <GRID, THREADS >> > (d_array, d_result, n);

    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    verify_result(h_array, h_mask, h_result, n);
}

int main() {

    int n = 1 << 20;

    int bytes_n = n * sizeof(int);
    size_t bytes_m = MASK_LENGTH * sizeof(int);

    int* h_array = new int[n];
    int* h_mask = new int[MASK_LENGTH];
    int* h_result = new int[n];

    int* d_array, * d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_result, bytes_n);
    
    initialize_vector(h_array, n);
    initialize_vector(h_mask, MASK_LENGTH);
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    int THREADS = 256;

    int GRID = (n + THREADS - 1) / THREADS;

    cout << "Normal Convolution" << endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        NormalConvolution(h_array, h_mask, h_result, d_array, d_result, n, THREADS, GRID, bytes_n);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "\nTime taken without CUDA GRAPH in ms : " << milliseconds / GRAPH_LAUNCH_ITERATIONS << endl;
    
    cout << "----------------------------------------------------" << endl;
    
    cout << "Convolution using CUDA GRAPHS (Graph API)" << endl;
    
    cudaGraphAPIconvolution(h_array, h_mask, h_result, d_array, d_result, n, THREADS, GRID, bytes_n, bytes_m);
    
    cout << "----------------------------------------------------\n" << endl;
    cout << "\nThe time is the average time of all the kernel launchs. The total kernel launches are " << GRAPH_LAUNCH_ITERATIONS << endl;
    
    
    delete[] h_array;
    delete[] h_result;
    delete[] h_mask;
    cudaFree(d_result);
    cudaFree(d_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}