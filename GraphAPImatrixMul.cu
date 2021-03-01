#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#define GRAPH_LAUNCH_ITERATIONS  300


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

void cudaGraphAPImatrixMul(int* h_a, int* h_b, int* h_c, int* d_a, int* d_b, int* d_c, dim3 threads, dim3 blocks, int N) {

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
    memcpyParams.srcPtr = make_cudaPitchedPtr(h_a, sizeof(int) * N * N, N * N, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(d_a, sizeof(int) * N * N, N * N, 1);
    memcpyParams.extent = make_cudaExtent(sizeof(int) * N * N, 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;

    //Adding memsetParams node
    memsetParams.dst = (void*)d_c;
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(int);
    memsetParams.width = N * N;
    memsetParams.height = 1;

    cudaGraphCreate(&graph, 0);
    cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams);
    cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);
    nodeDependencies.push_back(memsetNode);
    nodeDependencies.push_back(memcpyNode);

    //Adding memcpyParams node
    memset(&memcpyParams, 0, sizeof(memcpyParams));
    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(h_b, sizeof(int) * N * N, N * N, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(d_b, sizeof(int) * N * N, N * N, 1);
    memcpyParams.extent = make_cudaExtent(sizeof(int) * N * N, 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;

    cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams);
    nodeDependencies.push_back(memcpyNode);

    //Adding Kernal node
    void* kernelArgs[4] = { (void*)&d_a, (void*)&d_b, &d_c, &N };
    kernelNodeParams.func = (void*)matrixMul;
    kernelNodeParams.gridDim = blocks;
    kernelNodeParams.blockDim = threads;
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
    memcpyParams.srcPtr = make_cudaPitchedPtr(d_c, sizeof(int) * N * N, N * N, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(h_c, sizeof(int) * N * N, N * N, 1);
    memcpyParams.extent = make_cudaExtent(sizeof(int) * N * N, 1, 1);
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
        verify_result(h_a, h_b, h_c, N);
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
        verify_result(h_a, h_b, h_c, N);
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



void NormalMatrixMul(int* h_a, int* h_b, int* h_c, int* d_a, int* d_b, int* d_c, dim3 threads, dim3 blocks, int N) {

    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    matrixMul << <blocks, threads >> > (d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, N);

}


int main() {

    int N = 1 << 6;

    size_t bytes = N * N * sizeof(int);

    int* h_a, * h_b, * h_c;

    h_a = new int[N * N];
    h_b = new int[N * N];
    h_c = new int[N * N];

    init_matrix(h_a, N);
    init_matrix(h_b, N);

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

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
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "\nTime taken without CUDA GRAPH in ms : " << milliseconds / GRAPH_LAUNCH_ITERATIONS << endl;
    
    cout << "----------------------------------------------------" << endl;
    
    cout << "Matrix Mul using CUDA GRAPHS (Graph API)" << endl;
    
    cudaGraphAPImatrixMul(h_a, h_b, h_c, d_a, d_b, d_c, threads, blocks, N);

    cout << "----------------------------------------------------\n" << endl;
    cout << "\nThe time is the average time of all the kernel launchs. The total kernel launches are " << GRAPH_LAUNCH_ITERATIONS << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}