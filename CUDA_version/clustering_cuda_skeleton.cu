/* 
 * COMPILE: nvcc -std=c++11 clustering_cuda_skeleton.cu clustering_impl.cpp main.cpp -o cuda
 * RUN:     ./cuda <path> <epsilon> <mu> <num_blocks_per_grid> <num_threads_per_block>
 */

#include <iostream>
#include "clustering.h"

void expansion(int v, int label, int *nbr_offs, int *sim_nbrs, 
    bool *visited, bool *pivots, int *cluster_result){

    for (int j=nbr_offs[v]; j<nbr_offs[v+1]; j++){
        if (sim_nbrs[j] != -1){
            int nbr_id = sim_nbrs[j];
            if((pivots[nbr_id]) && (!visited[nbr_id])){
                visited[nbr_id] = true;
                cluster_result[nbr_id] = label;
                expansion(nbr_id, label, nbr_offs, 
                    sim_nbrs, visited, pivots, cluster_result);
            }
        } else{
            break;
        }
    }      
}

__global__ void parallel(int *nbr_offs_device, int *nbrs_device, float epsilon, int mu, 
                         int num_vs, bool *pivots_device, int *sim_nbrs_device){
    
    //Stage 1
    const int num_thread = blockDim.x*gridDim.x;

    for(int i = blockDim.x*blockIdx.x + threadIdx.x; i<num_vs; i+=num_thread){
        // i: Vertex id
        int left_start = nbr_offs_device[i];
        int left_end = nbr_offs_device[i+1];
        int left_size = left_end - left_start;
        int counter = 0;

        for (int j = left_start; j < left_end; j++){
            int nbr_id = nbrs_device[j];

            int right_start = nbr_offs_device[nbr_id];
            int right_end = nbr_offs_device[nbr_id + 1];
            int right_size = right_end - right_start;

            // compute the similarity
            int left_pos = left_start, right_pos = right_start, num_com_nbrs = 0;
    
            while (left_pos < left_end && right_pos < right_end) {
                if (nbrs_device[left_pos] == nbrs_device[right_pos]) {
                    num_com_nbrs++;
                    left_pos++;
                    right_pos++;
                } else if (nbrs_device[left_pos] < nbrs_device[right_pos]) {
                    left_pos++;
                } else {
                    right_pos++;
                }
            }

            float sim = (num_com_nbrs + 2) / sqrt((left_size + 1.0) * (right_size + 1.0));

            if (sim > epsilon) {
                sim_nbrs_device[nbr_offs_device[i] + counter]  = nbr_id;
                counter++;
            }
        }
        if(counter > mu) pivots_device[i] = true;      
    }
}

void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {

    // Stage 1
    bool *pivots, *pivots_device, *visited;
    int *nbrs_device, *nbr_offs_device;
    int *sim_nbrs, *sim_nbrs_device;

    pivots = (bool *)malloc(num_vs * sizeof(bool));
    visited = (bool *)malloc(num_vs * sizeof(bool)); 
    sim_nbrs = (int *)malloc(num_es * sizeof(int));

    size_t size_vs_bool = num_vs * sizeof(bool);
    size_t size_vs_int = (num_vs+1) * sizeof(int);
    size_t size_es_int = (num_es+1) * sizeof(int);


    cudaMalloc(&pivots_device, size_vs_bool);
    cudaMalloc(&nbrs_device, size_es_int);
    cudaMalloc(&nbr_offs_device, size_vs_int);
    cudaMalloc(&sim_nbrs_device, size_es_int);
    cudaMemset(sim_nbrs_device, -1, size_es_int);
    cudaMemset(pivots_device, false, size_vs_bool);

    std::fill(cluster_result, cluster_result + num_vs, -1);
    std::fill(visited, visited + num_vs, false);
    
    cudaMemcpy(nbr_offs_device, nbr_offs, size_vs_int, cudaMemcpyHostToDevice);
    cudaMemcpy(nbrs_device, nbrs, size_es_int, cudaMemcpyHostToDevice);

    parallel<<<num_blocks_per_grid, num_threads_per_block>>>(nbr_offs_device, nbrs_device,
        epsilon, mu, num_vs, pivots_device, sim_nbrs_device);
    
    cudaMemcpy(pivots, pivots_device, size_vs_bool, cudaMemcpyDeviceToHost);
    cudaMemcpy(sim_nbrs, sim_nbrs_device, size_es_int, cudaMemcpyDeviceToHost);

    // Stage 2
    for (int i = 0; i < num_vs; i++) {
        if (!pivots[i] || visited[i]) continue;

        visited[i] = true;
        cluster_result[i] = i;
        expansion(i, i, nbr_offs, sim_nbrs, visited, pivots, cluster_result);
    }

    num_clusters = 0;
    for (auto i = 0; i< num_vs; i++){
        if (cluster_result[i] == i)
            num_clusters++;
    }

    free(pivots);
    free(visited);
    free(sim_nbrs);

    cudaFree(pivots_device);
    cudaFree(nbrs_device);
    cudaFree(nbr_offs_device);
}
