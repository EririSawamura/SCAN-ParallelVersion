/* 
  usage:
  mpic++ -std=c++11 clustering_mpi_skeleton.cpp clustering_impl.cpp -o clustering
  mpiexec -n <num_processes> ./clustering <test_folder> <result_folder>

  e.g. mpiexec -n 4 ./clustering ./dataset/test1 ./results/
*/

#include "clustering.h"

#include "mpi.h"

#include <cassert>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm;
  int num_process; // number of processors
  int my_rank;     // my global rank

  comm = MPI_COMM_WORLD;

  MPI_Comm_size(comm, &num_process);
  MPI_Comm_rank(comm, &my_rank);

  if (argc != 3) {
    std::cerr << "usage: ./clustering_sequential data_path result_path"
              << std::endl;

    return -1;
  }
  std::string dir(argv[1]);
  std::string result_path(argv[2]);

  int num_graphs;
  int *clustering_results = nullptr;
  int *num_cluster_total = nullptr;

  int *nbr_offs = nullptr, *nbrs = nullptr;
  int *nbr_offs_local = nullptr, *nbrs_local = nullptr;

  GraphMetaInfo *info = nullptr;

  // read graph info from files
  if (my_rank == 0) {
    num_graphs = read_files(dir, info, nbr_offs, nbrs);
  }
  auto start_clock = chrono::high_resolution_clock::now();

  // ADD THE CODE HERE

  int *clustering_results_locals, *num_cluster_local, *infos;
  int *info_locals, *nbr_scounts, *nbr_displs, *nbrs_scounts, *nbrs_displs, *cr_displs;
  int p_process, *recv_count;

  if (my_rank == 0){
    num_cluster_total = (int *)calloc(num_graphs, sizeof(int));
    p_process = num_graphs / num_process;
    num_cluster_local = (int *)calloc(p_process, sizeof(int));
    infos = (int *)calloc(num_graphs * 2, sizeof(int));
    nbr_scounts = (int *)calloc(num_process, sizeof(int));
    nbr_displs = (int *)calloc(num_process, sizeof(int));
    nbrs_scounts = (int *)calloc(num_process, sizeof(int));
    nbrs_displs = (int *)calloc(num_process, sizeof(int));
    recv_count = (int *)calloc(num_process, sizeof(int));
    cr_displs = (int *)calloc(num_process, sizeof(int));

    info_locals = (int *)calloc(p_process*2, sizeof(int));

    int recv_nbr_count = 0;
    int recv_nbrs_count = 0;
    int recv_total = 0;

    for(int i=0; i<num_process; i++){
      nbr_scounts[i] = 0;
      nbr_displs[i] = 0;
      nbrs_scounts[i] = 0;
      nbrs_displs[i] = 0;
      if (i!=0){
        MPI_Send(&num_graphs, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
    }

    for(int i=0; i<num_graphs; i++){
      infos[2*i] = info[i].num_vertices;
      infos[2*i+1] = info[i].num_edges;

      nbr_scounts[i / p_process] += info[i].num_vertices + 1;
      nbrs_scounts[i / p_process] += info[i].num_edges + 1;
      recv_count[i / p_process] += info[i].num_vertices;
      recv_total += info[i].num_vertices;
    }

    for(int i=0; i<num_process; i++){
      for(int j=0; j<i; j++){
        nbrs_displs[i] += nbrs_scounts[j];
        nbr_displs[i] += nbr_scounts[j];
        cr_displs[i] += recv_count[j];
      }
    }

    MPI_Scatter(infos, 2*p_process, MPI_INT, info_locals, 2*p_process, MPI_INT, 0, MPI_COMM_WORLD);

    for(int i=0; i<p_process; i++){
      recv_nbr_count += info_locals[2*i] + 1;
      recv_nbrs_count += info_locals[2*i+1] + 1;
    }
    nbr_offs_local = (int *)calloc(recv_nbr_count, sizeof(int));
    nbrs_local = (int *)calloc(recv_nbrs_count, sizeof(int));
    clustering_results_locals = (int *)calloc(recv_nbr_count-p_process, sizeof(int));
    clustering_results = (int *)calloc(recv_total, sizeof(int));

    MPI_Scatterv(nbrs, nbrs_scounts, nbrs_displs, MPI_INT, nbrs_local, recv_nbrs_count, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Scatterv(nbr_offs, nbr_scounts, nbr_displs, MPI_INT, nbr_offs_local, recv_nbr_count, MPI_INT, 0, MPI_COMM_WORLD );
    free(infos);

    int *temp = clustering_results_locals;
    for(int j=0; j < p_process; j++){
      GraphMetaInfo info_local;
      info_local.num_vertices = info_locals[2*j];
      info_local.num_edges = info_locals[2*j+1];

      num_cluster_local[j] = clustering(info_local, nbr_offs_local, nbrs_local,
                                      temp);
      nbr_offs_local += (info_local.num_vertices + 1);
      nbrs_local += (info_local.num_edges + 1);
      temp += info_local.num_vertices;
    }

    MPI_Gatherv(clustering_results_locals, recv_nbr_count-p_process, MPI_INT, clustering_results, recv_count, cr_displs, MPI_INT, 0, MPI_COMM_WORLD);

  } else {
    //other processes
    MPI_Recv(&num_graphs, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    p_process = num_graphs / num_process;
    info_locals = (int *)calloc(p_process*2, sizeof(int));

    int recv_nbr_count = 0;
    int recv_nbrs_count = 0;

    MPI_Scatter(infos, 2*p_process, MPI_INT, info_locals, 2*p_process, MPI_INT, 0, MPI_COMM_WORLD);
    
    for(int i=0; i<p_process; i++){
      recv_nbr_count += info_locals[2*i] + 1;
      recv_nbrs_count += info_locals[2*i+1] + 1;
    }
    nbr_offs_local = (int *)calloc(recv_nbr_count, sizeof(int));
    nbrs_local = (int *)calloc(recv_nbrs_count, sizeof(int));
    clustering_results_locals = (int *)calloc(recv_nbr_count-p_process, sizeof(int));

    MPI_Scatterv(nbrs, nbrs_scounts, nbrs_displs, MPI_INT, nbrs_local, recv_nbrs_count, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Scatterv(nbr_offs, nbr_scounts, nbr_displs, MPI_INT, nbr_offs_local, recv_nbr_count, MPI_INT, 0, MPI_COMM_WORLD );

    num_cluster_local = (int *)calloc(p_process, sizeof(int));
    int *temp = clustering_results_locals;

    for(int j=0; j<p_process; j++){
      GraphMetaInfo info_local;
      info_local.num_vertices = info_locals[2*j];
      info_local.num_edges = info_locals[2*j+1];

      num_cluster_local[j] = clustering(info_local, nbr_offs_local, nbrs_local,
                                          temp);
      nbr_offs_local += (info_local.num_vertices + 1);
      nbrs_local += (info_local.num_edges + 1);
      temp += info_local.num_vertices;
    }
    
    MPI_Gatherv(clustering_results_locals, recv_nbr_count-p_process, MPI_INT, clustering_results, recv_count, cr_displs, MPI_INT, 0, MPI_COMM_WORLD);
  }

  MPI_Gather(num_cluster_local, p_process, MPI_INT, num_cluster_total, p_process, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(comm);
  auto end_clock = chrono::high_resolution_clock::now();

  // 1) print results to screen
  if (my_rank == 0) {
    for (size_t i = 0; i < num_graphs; i++) {
      printf("num cluster in graph %d : %d\n", i, num_cluster_total[i]);
    }
    fprintf(stderr, "Elapsed Time: %.9lf ms\n",
            chrono::duration_cast<chrono::nanoseconds>(end_clock - start_clock)
                    .count() /
                pow(10, 6));
  }

  // 2) write results to file
  if (my_rank == 0) {
    int *result_graph = clustering_results;
    for (int i = 0; i < num_graphs; i++) {
      GraphMetaInfo info_local = info[i];
      write_result_to_file(info_local, i, num_cluster_total[i], result_graph,
                           result_path);

      result_graph += info_local.num_vertices;
    }
  }

  MPI_Finalize();

  if (my_rank == 0) {
    free(num_cluster_total);
  }

  return 0;
}