/*
 *
 * Please only change this file and do not change any other files.
 * Feel free to change/add any helper functions.
 *
 * COMPILE: g++ -lstdc++ -std=c++11 -pthread clustering_pthread_skeleton.cpp main.cpp -o pthread
 * RUN:     ./pthread_test datasets/test1/1.txt 0.7 3 4
 */

#include <pthread.h>
#include "clustering.h"
#include <math.h>
#include <chrono>

int vs, es;
int *nbr_off, *nbr, *cluster_result;
vector <int> *nbre;
float g_epsilon, g_mu;
bool *pivots, *visited;
pthread_mutex_t mutex;

struct AllThings{
    int num_threads;
    int my_rank;

    AllThings(int inum_threads, int imy_rank){
        num_threads = inum_threads;
        my_rank = imy_rank;
    };
};

void expansion(int v, int label){
    vector<int>::iterator it_i;
    for(it_i=nbre[v].begin(); it_i!=nbre[v].end(); ++it_i){
        int nbr_id = *it_i;
        if((pivots[nbr_id]) && (!visited[nbr_id])){
            visited[nbr_id] = true;
            cluster_result[nbr_id] = label;
            expansion(nbr_id, label);
        }
    }
}

void *parallel(void* allthings){
    AllThings *all = (AllThings *) allthings;
    int rank = all->my_rank, num_threads = all->num_threads;
    
    //Stage 1
    if (rank == num_threads - 1){
        for(int i=rank*(vs / num_threads + 1); i<vs; i++){
            // i: Vertex id
            int *left_start = &nbr[nbr_off[i]];
            int *left_end = &nbr[nbr_off[i + 1]];
            int left_size = left_end - left_start;

            for (int *j = left_start; j < left_end; j++){
                int nbr_id = *j;

                int *right_start = &nbr[nbr_off[nbr_id]];
                int *right_end = &nbr[nbr_off[nbr_id + 1]];
                int right_size = right_end - right_start;

                // compute the similarity
                int num_com_nbrs = get_num_com_nbrs(left_start, left_end, right_start, right_end);

                float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

                if (sim > g_epsilon) {
                    nbre[i].push_back(nbr_id);
                }
            } 
            if(nbre[i].size() > g_mu) pivots[i] = true;
        } 
    } else {
        for(int i=rank*(vs / num_threads + 1); i<(rank+1)*(vs / num_threads + 1); i++){
            // i: Vertex id
            int *left_start = &nbr[nbr_off[i]];
            int *left_end = &nbr[nbr_off[i + 1]];
            int left_size = left_end - left_start;

            for (int *j = left_start; j < left_end; j++){
                int nbr_id = *j;

                int *right_start = &nbr[nbr_off[nbr_id]];
                int *right_end = &nbr[nbr_off[nbr_id + 1]];
                int right_size = right_end - right_start;

                // compute the similarity
                int num_com_nbrs = get_num_com_nbrs(left_start, left_end, right_start, right_end);

                float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

                if (sim > g_epsilon) {
                    nbre[i].push_back(nbr_id);
                }
            } 
            if(nbre[i].size() > g_mu) pivots[i] = true;   
        } 
    }
    return 0;
}

int *scan(float epsilon, int mu, int num_threads, int num_vs, int num_es, int *nbr_offs, int *nbrs){
    long thread;
    vs = num_vs;
    es = num_es;
    nbr_off = nbr_offs;
    nbr = nbrs;
    g_epsilon = epsilon;
    g_mu = mu;
    pivots = (bool *)malloc(num_vs * sizeof(bool));
    visited = (bool *)malloc(num_vs * sizeof(bool)); 
    nbre= new vector<int>[num_vs];
    pthread_t* thread_handles = (pthread_t*) malloc(num_threads*sizeof(pthread_t));
    cluster_result = new int[num_vs];
    pthread_mutex_init(&mutex, NULL);

    std::fill(cluster_result, cluster_result + num_vs, -1);
    std::fill(pivots, pivots + num_vs, false);
    std::fill(visited, visited + num_vs, false);

    for (thread=0; thread < num_threads; thread++)
        pthread_create(&thread_handles[thread], NULL, parallel, 
            (void *) new AllThings(num_threads, thread));
    for (thread=0; thread < num_threads; thread++)
        pthread_join(thread_handles[thread], NULL);
    
    // Stage 2:
    int num_clusters = 0;
    for (int i = 0; i < num_vs; i++) {
        if (!pivots[i] || visited[i]) continue;

        visited[i] = true;
        cluster_result[i] = i;
        expansion(i, i);

        num_clusters++;
    }

    pthread_mutex_destroy(&mutex);
    free(pivots);
    return cluster_result;
}