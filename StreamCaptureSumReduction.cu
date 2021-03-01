#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define SIZE 1024
#define SHMEM_SIZE 1024*sizeof(int)
#define GRAPH_LAUNCH_ITERATIONS  3

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

void cudaGraphsumReduction(int* h_v, int* h_v_r, int* d_v, int* d_v_r, int TB_SIZE, int GRID_SIZE, int n) {

	cudaStream_t stream1, stream2, streamForGraph;
	cudaEvent_t forkStreamEvent, memsetEvent;
	cudaGraph_t graph;
	int result = 0;

	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking);

	cudaEventCreate(&forkStreamEvent);
	cudaEventCreate(&memsetEvent);

	cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
	cudaEventRecord(forkStreamEvent, stream1);
	cudaStreamWaitEvent(stream2, forkStreamEvent);

	cudaMemcpyAsync(d_v, h_v, n * sizeof(int), cudaMemcpyDefault, stream1);

	cudaMemsetAsync(d_v_r, 0, n * sizeof(int), stream2);
	cudaEventRecord(memsetEvent, stream2);
	cudaStreamWaitEvent(stream1, memsetEvent);

	sum_reduction << <GRID_SIZE, TB_SIZE, 0, stream1 >> > (d_v, d_v_r);

	sum_reduction << <1, TB_SIZE, 0, stream1 >> > (d_v_r, d_v_r);

	cudaMemcpyAsync(h_v_r, d_v_r, n * sizeof(int), cudaMemcpyDefault, stream1);

	cudaStreamEndCapture(stream1, &graph);

	cudaGraphNode_t* nodes = NULL;
	size_t numNodes = 0;
	cudaGraphGetNodes(graph, nodes, &numNodes);
	cout << "\nNum of nodes in the graph created using stream capture API = " << numNodes << endl;

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
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	cout << "\nVerifying Cloned Graph ... " << endl;
	for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++)
	{
		cudaGraphLaunch(clonedGraphExec, streamForGraph);
		cudaStreamSynchronize(streamForGraph);
		verify_result(h_v, h_v_r, n);
	}
	cout << "Done! Verifyied successfully" << endl;

	cout << "\nTime taken by using CUDA GRAPH in ms : " << milliseconds / GRAPH_LAUNCH_ITERATIONS << endl;

	cudaStreamSynchronize(streamForGraph);

	cudaGraphExecDestroy(graphExec);
	cudaGraphExecDestroy(clonedGraphExec);
	cudaGraphDestroy(graph);
	cudaGraphDestroy(clonedGraph);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
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

	cout << "Normal Sum Reduction" << endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
		sumReduction(h_v, h_v_r, d_v, d_v_r, TB_SIZE, GRID_SIZE, n);
		cudaStreamSynchronize(0);
	}

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "\nTime taken without CUDA GRAPH in ms : " << milliseconds / GRAPH_LAUNCH_ITERATIONS << endl;

	cout << "----------------------------------------------------" << endl;
	
	cout << "Sum Reduction using CUDA GRAPHS (Stream Capture)" << endl;
	cudaGraphsumReduction(h_v, h_v_r, d_v, d_v_r, TB_SIZE, GRID_SIZE, n);

	cout << "----------------------------------------------------\n" << endl;
	cout << "\nThe time is the average time of all the kernel launchs. The total kernel launches are " << GRAPH_LAUNCH_ITERATIONS << endl;

	cudaFree(d_v);
	cudaFree(d_v_r);
	return 0;
}

