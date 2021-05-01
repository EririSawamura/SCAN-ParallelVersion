/* 
 * COMPILE: nvcc -std=c++11 clustering_cuda_skeleton.cu clustering_impl.cpp main.cpp -o cuda
 * RUN:     ./cuda <path> <epsilon> <mu> <num_blocks_per_grid> <num_threads_per_block>
 *          ./cuda ./datasets/test1/1.txt 0.7 3 8 512
 */

#include <iostream>
#include "clustering.h"

// Define variables or functions here
const int TILE_WIDTH;
vector <int> *sim_nbrs_v;

void expansion(int v, int label){
    vector<int>::iterator it_i;
    for(it_i=sim_nbrs_v[v].begin(); it_i!=sim_nbrs_v[v].end(); ++it_i){
        int nbr_id = *it_i;
        if((pivots[nbr_id]) && (!visited[nbr_id])){
            visited[nbr_id] = true;
            cluster_result[nbr_id] = label;
            expansion(nbr_id, label);
        }
    }
}

__global__ void parallel(int *nbr_offs_device, int *nbrs_device, float epsilon, int mu, 
                        int num_vs, int num_es, bool *pivots_device,
                        int num_blocks_per_grid, int num_threads_per_block, 
                        int *sim_nbrs_device){
    
    //Stage 1
    int bx = blockIdx.x, tx = threadIdx.x, vs;
    int counter = 0;
    if (bx * num_threads_per_block * TILE_WIDTH + tx * TILE_WIDTH < num_vs - TILE_WIDTH){
        vs = TILE_WIDTH;
    } else {
        vs = num_vs - (bx * num_threads_per_block * TILE_WIDTH + tx * TILE_WIDTH);
    }

    for(int i = bx * num_threads_per_block * TILE_WIDTH + tx * TILE_WIDTH; i<vs; i++){
        // i: Vertex id
        int *left_start = &nbrs_device[nbr_offs_device[i]];
        int *left_end = &nbrs_device[nbr_offs_device[i+1]];
        int left_size = left_end - left_start;

        for (int *j = left_start; j < left_end; j++){
            int nbr_id = *j;

            int *right_start = &nbrs_device[nbr_offs_device[nbr_id]];
            int *right_end = &nbrs_device[nbr_offs_device[nbr_id + 1]];
            int right_size = right_end - right_start;

            // compute the similarity
            int num_com_nbrs = get_num_com_nbrs(left_start, left_end, right_start, right_end);

            float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

            if (sim > g_epsilon) {
                sim_nbrs_device[nbr_offs_device[i] + counter]  = nbr_id;
                counter++;
            }
        } 
        if(counter > g_mu) pivots_device[i] = true;
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
    sim_nbrs_v = new vector<int>[num_vs];

    TILE_WIDTH = num_vs / (num_threads_per_block * num_blocks_per_grid);
    cudaMalloc(&pivots_device, num_vs);
    cudaMalloc(&nbrs_device, num_es);
    cudaMalloc(&nbr_offs_device, num_vs);
    cudaMalloc(&sim_nbrs_device, num_es);
    cudaMemset(sim_nbrs_device, -1, num_es);

    std::fill(cluster_result, cluster_result + num_vs, -1);
    std::fill(pivots, pivots + num_vs, false);
    std::fill(visited, visited + num_vs, false);
    
    parallel<<<num_blocks_per_grid, num_threads_per_block>>>(nbr_offs_device, nbrs_device,
        epsilon, mu, num_vs, num_es, pivots_device, num_blocks_per_grid, 
        num_threads_per_block, sim_nbrs_device);
    
    cudaMemcpy(pivots, pivots_device, num_vs, cudaMemcpyDeviceToHost);
    cudaMemcpy(sim_nbrs, sim_nbrs_device, num_es, cudaMemcpyDeviceToHost);
    
    for (int i=0; i<num_vs; i++) {
        if (i == num_vs - 1){
            for (int j=nbr_offs[i]; j<num_es; j++){
                if (sim_nbrs[j] != -1){
                    sim_nbrs_v[i].add(sim_nbrs[j])
                } else{
                    break;
                }
            }
        } else{
            for (int j=nbr_offs[i]; j<nbr_offs[i+1]; j++){
                if (sim_nbrs[j] != -1){
                    sim_nbrs_v[i].add(sim_nbrs[j])
                } else{
                    break;
                }
            }
        }        
    }

    // Stage 2
    for (int i = 0; i < num_vs; i++) {
        if (!pivots[i] || visited[i]) continue;

        visited[i] = true;
        cluster_result[i] = i;
        expansion(i, i);
    }

    num_clusters = 0;
    for (auto i = 0; i< num_vs; i++){
        if (cluster_result[i] == i)
            num_clusters++;
    }

    free(pivots);
    free(visited);
    free(sim_nbrs);
    free(sim_nbrs_v);

    cudaFree(pivots_device);
    cudaFree(nbrs_device);
    cudaFree(nbr_offs_device);
    cudaFree(sim_nbrs_device);
}
