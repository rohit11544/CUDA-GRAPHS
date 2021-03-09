#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#define GRAPH_LAUNCH_ITERATIONS  3


using std::endl;
using std::cout;


void init_matrix(int* a, const int N) {
    for (int i = 0; i < N * N; i++) {
        a[i] = rand() % 100;
    }
}

__global__ void matrixMul(int* a, int* b, int* c, const int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

void verify_result(int* a, int* b, int* c, const int N) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }
            if (tmp != c[i * N + j]) {
                cout << "NOT SUCCESSFUL" << endl;
                exit(0);
            }
        }
    }
}



void StreamCaptureMatrixMul(int* h_a, int* h_b, int* h_c, int* d_a, int* d_b, int* d_c, dim3 threads, dim3 blocks, int N) {

    cudaStream_t stream1, stream2, stream3, streamForGraph;
    cudaEvent_t forkStreamEvent, memcpyEvent, memsetEvent;
    cudaGraph_t graph;
    int result = 0;

    checkCudaErrors(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking));

    checkCudaErrors(cudaEventCreate(&forkStreamEvent));
    checkCudaErrors(cudaEventCreate(&memcpyEvent));
    checkCudaErrors(cudaEventCreate(&memsetEvent));

    checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));
    checkCudaErrors(cudaEventRecord(forkStreamEvent, stream1));
    checkCudaErrors(cudaStreamWaitEvent(stream2, forkStreamEvent));
    checkCudaErrors(cudaStreamWaitEvent(stream3, forkStreamEvent));

    checkCudaErrors(cudaMemcpyAsync(d_a, h_a, N * N * sizeof(int), cudaMemcpyDefault, stream1));
    checkCudaErrors(cudaMemcpyAsync(d_b, h_b, N * N * sizeof(int), cudaMemcpyDefault, stream2));
    checkCudaErrors(cudaEventRecord(memcpyEvent, stream2));
    checkCudaErrors(cudaStreamWaitEvent(stream1, memcpyEvent));

    checkCudaErrors(cudaMemsetAsync(d_c, 0, N * N * sizeof(int), stream3));
    checkCudaErrors(cudaEventRecord(memsetEvent, stream3));
    checkCudaErrors(cudaStreamWaitEvent(stream1, memsetEvent));

    matrixMul << <blocks, threads, 0, stream1 >> > (d_a, d_b, d_c, N);

    checkCudaErrors(cudaMemcpyAsync(h_c, d_c, N * N * sizeof(int), cudaMemcpyDefault, stream1));

    checkCudaErrors(cudaStreamEndCapture(stream1, &graph));

    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
    cout << "Num of nodes in the graph created using stream capture API = " << numNodes << endl;

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    cudaGraph_t clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
    checkCudaErrors(cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++)
    {
        cudaGraphLaunch(graphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
        verify_result(h_a, h_b, h_c, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "\nVerifying Cloned Graph ... " << endl;
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++)
    {
        checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
        checkCudaErrors(cudaStreamSynchronize(streamForGraph));
        verify_result(h_a, h_b, h_c, N);
    }
    cout << "Done! Verifyied successfully" << endl;

    cout << "\nTime taken by using CUDA GRAPH in ms : " << milliseconds / GRAPH_LAUNCH_ITERATIONS << endl;

    checkCudaErrors(cudaStreamSynchronize(streamForGraph));
    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaGraphDestroy(clonedGraph));
    checkCudaErrors(cudaStreamDestroy(stream1));
    checkCudaErrors(cudaStreamDestroy(stream2));
    checkCudaErrors(cudaStreamDestroy(streamForGraph));
    checkCudaErrors(cudaEventDestroy(memcpyEvent));
    checkCudaErrors(cudaEventDestroy(memsetEvent));
    checkCudaErrors(cudaEventDestroy(forkStreamEvent));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

}



void NormalMatrixMul(int* h_a, int* h_b, int* h_c, int* d_a, int* d_b, int* d_c, dim3 threads, dim3 blocks, int N) {

    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    matrixMul << <blocks, threads >> > (d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, N);

}


int main() {

    int N = 1 << 10;

    size_t bytes = N * N * sizeof(int);

    int* h_a, * h_b, * h_c;

    h_a = new int[N * N];
    h_b = new int[N * N];
    h_c = new int[N * N];

    int* d_a, * d_b, * d_c;
    checkCudaErrors(cudaMalloc(&d_a, bytes));
    checkCudaErrors(cudaMalloc(&d_b, bytes));
    checkCudaErrors(cudaMalloc(&d_c, bytes));

    init_matrix(h_a, N);
    init_matrix(h_b, N);

    int THREADS = 32;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    cout << "Normal Matrix Mul" << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        NormalMatrixMul(h_a, h_b, h_c, d_a, d_b, d_c, threads, blocks, N);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\nTime taken without CUDA GRAPH in ms : " << milliseconds / GRAPH_LAUNCH_ITERATIONS << endl;

    cout << "----------------------------------------------------" << endl;

    cout << "Matrix Mul using CUDA GRAPHS (Stream Capture)" << endl;

    StreamCaptureMatrixMul(h_a, h_b, h_c, d_a, d_b, d_c, threads, blocks, N);

    cout << "----------------------------------------------------\n" << endl;
    cout << "\nThe time is the average time of all the kernel launchs. The total kernel launches are " << GRAPH_LAUNCH_ITERATIONS << endl;

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}
