#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

#define SIZE 1024
#define SHMEM_SIZE 1024*sizeof(int)
#define GRAPH_LAUNCH_ITERATIONS  300

using std::cout;
using std::endl;

__global__ void sum_reduction(int* v, int* v_r) {
	__shared__ int partial_sum[SHMEM_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = rand() % 100;
	}
}

void verify_result(int* a, int* r, const int N) {
	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += a[i];
	}
	if (sum != r[0]) {
		cout << "NOT SUCCESSFUL" << endl;
		exit(0);
	}
}

void cudaGraphAPIsumReduction(int* h_v, int* h_v_r, int* d_v, int* d_v_r, int TB_SIZE, int GRID_SIZE, int n) {

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
	memcpyParams.srcPtr = make_cudaPitchedPtr(h_v, sizeof(int) * n, n, 1);
	memcpyParams.dstArray = NULL;
	memcpyParams.dstPos = make_cudaPos(0, 0, 0);
	memcpyParams.dstPtr = make_cudaPitchedPtr(d_v, sizeof(int) * n, n, 1);
	memcpyParams.extent = make_cudaExtent(sizeof(int) * n, 1, 1);
	memcpyParams.kind = cudaMemcpyHostToDevice;

	//Adding memsetParams node
	memsetParams.dst = (void*)d_v_r;
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
	void* kernelArgs[2] = { (void*)&d_v, (void*)&d_v_r };
	kernelNodeParams.func = (void*)sum_reduction;
	kernelNodeParams.gridDim = dim3(GRID_SIZE, 1, 1);
	kernelNodeParams.blockDim = dim3(TB_SIZE, 1, 1);
	kernelNodeParams.sharedMemBytes = 0;
	kernelNodeParams.kernelParams = (void**)kernelArgs;
	kernelNodeParams.extra = NULL;

	cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);
	nodeDependencies.clear();
	nodeDependencies.push_back(kernelNode);

	//Adding Kernal node
	memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
	void* kernelArgs2[2] = { (void*)&d_v_r, (void*)&d_v_r };
	kernelNodeParams.func = (void*)sum_reduction;
	kernelNodeParams.gridDim = dim3(1, 1, 1);
	kernelNodeParams.blockDim = dim3(TB_SIZE, 1, 1);
	kernelNodeParams.sharedMemBytes = 0;
	kernelNodeParams.kernelParams = (void**)kernelArgs2;
	kernelNodeParams.extra = NULL;

	cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);
	nodeDependencies.clear();
	nodeDependencies.push_back(kernelNode);

	//Adding memcpyParams node
	memset(&memcpyParams, 0, sizeof(memcpyParams));
	memcpyParams.srcArray = NULL;
	memcpyParams.srcPos = make_cudaPos(0, 0, 0);
	memcpyParams.srcPtr = make_cudaPitchedPtr(d_v_r, sizeof(int) * n, n, 1);
	memcpyParams.dstArray = NULL;
	memcpyParams.dstPos = make_cudaPos(0, 0, 0);
	memcpyParams.dstPtr = make_cudaPitchedPtr(h_v_r, sizeof(int) * n, n, 1);
	memcpyParams.extent = make_cudaExtent(sizeof(int) * n, 1, 1);
	memcpyParams.kind = cudaMemcpyDeviceToHost;

	cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams);
	nodeDependencies.clear();
	nodeDependencies.push_back(memcpyNode);

	cudaGraphNode_t* nodes = NULL;
	size_t numNodes = 0;
	cudaGraphGetNodes(graph, nodes, &numNodes);
	cout << "Num of nodes in the graph created manually = " << numNodes << endl;

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
		verify_result(h_v, h_v_r, n);
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
		verify_result(h_v, h_v_r, n);
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



void sumReduction(int* h_v, int* h_v_r, int* d_v, int* d_v_r, int TB_SIZE, int GRID_SIZE, int n) {

	cudaMemcpy(d_v, h_v, n * sizeof(int), cudaMemcpyHostToDevice);

	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);

	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	cudaMemcpy(h_v_r, d_v_r, n * sizeof(int), cudaMemcpyDeviceToHost);

	verify_result(h_v, h_v_r, n);

}

int main() {

	int n = 1 << 20;
	size_t bytes = n * sizeof(int);

	int* h_v, * h_v_r;
	int* d_v, * d_v_r;

	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	initialize_vector(h_v, n);

	int	TB_SIZE = SIZE;
	int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE;
	
	cout << "Normal Sum Reduction\n" << endl;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
		sumReduction(h_v, h_v_r, d_v, d_v_r, TB_SIZE, GRID_SIZE, n);
		cudaStreamSynchronize(0);
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "Time taken without CUDA GRAPH in ms : " << milliseconds/ GRAPH_LAUNCH_ITERATIONS << endl;

	cout << "----------------------------------------------------\n" << endl;

	cout << "Sum Reduction using CUDA GRAPHS (Graph API)\n" << endl;

	cudaGraphAPIsumReduction(h_v, h_v_r, d_v, d_v_r, TB_SIZE, GRID_SIZE, n);
	
	cout << "----------------------------------------------------\n" << endl;
	cout << "\nThe time is the average time of all the kernel launchs. The total kernel launches are " << GRAPH_LAUNCH_ITERATIONS << endl;
	
	cudaFree(d_v);
	cudaFree(d_v_r);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}

