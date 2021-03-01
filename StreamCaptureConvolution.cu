#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
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

void StreamCaptureConvolution(int* h_array, int* h_mask, int* h_result, int* d_array, int* d_result, const int n, int THREADS, int GRID, size_t bytes_n, size_t bytes_m) {

    cudaStream_t stream1, stream2, streamForGraph;
    cudaEvent_t forkStreamEvent, memcpyEvent;
    cudaGraph_t graph;
    int result = 0;

    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking);

    cudaEventCreate(&forkStreamEvent);
    cudaEventCreate(&memcpyEvent);

    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
    cudaEventRecord(forkStreamEvent, stream1);
    cudaStreamWaitEvent(stream2, forkStreamEvent);

    cudaMemcpyAsync(d_array, h_array, bytes_n, cudaMemcpyDefault, stream2);
    cudaEventRecord(memcpyEvent, stream2);
    cudaStreamWaitEvent(stream1, memcpyEvent);

    convolution_1d << <GRID, THREADS, 0, stream1 >> > (d_array, d_result, n);

    cudaMemcpyAsync(h_result, d_result, bytes_n, cudaMemcpyDefault, stream1);

    cudaStreamEndCapture(stream1, &graph);

    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nodes, &numNodes);
    cout << "Num of nodes in the graph created using stream capture API = " << numNodes << endl;

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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "\nVerifying Cloned Graph ... " << endl;
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
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(streamForGraph);
    cudaEventDestroy(forkStreamEvent);
    cudaEventDestroy(memcpyEvent);
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
        cudaStreamSynchronize(0);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\nTime taken without CUDA GRAPH in ms : " << milliseconds / GRAPH_LAUNCH_ITERATIONS << endl;
    
    cout << "----------------------------------------------------" << endl;
    
    cout << "Convolution using CUDA GRAPHS (Stream Capture)" << endl;
    
    StreamCaptureConvolution(h_array, h_mask, h_result, d_array, d_result, n, THREADS, GRID, bytes_n, bytes_m);

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