#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "dd.h"
#include "Pdd.h"
#include "mc_mpi_data.hpp"
#include <dolfin/function/Expression.h>

#include <mpi.h>

// multithread cpu entry point
std::vector<double> montecarlo(double *D, int dim, double* node_coord, int nof_nodes,
                               std::shared_ptr<const dolfin::Expression> f, std::shared_ptr< const dolfin::Expression> q,
                               const int walks, const double btol, const int threads, const enum impl_t impl)
{
    std::vector<double> est;

    if (dim == 2) {
        struct pdd_prm<2> pp;
        pp.max_nof_threads = threads;
        pp.D[0] = D[0];
        pp.D[1] = D[1];
        pp.nof_walks = walks;
        pp.btol = btol;
        pp.impl = impl;
        Pdd<2> pdd(pp,f,q);
        est = pdd.pdd(nof_nodes,node_coord);
    }
    else {
        struct pdd_prm<3> pp;
        pp.max_nof_threads = threads;
        pp.D[0] = D[0];
        pp.D[1] = D[1];
        pp.D[2] = D[2];
        pp.nof_walks = walks;
        pp.btol = btol;
        pp.impl = impl;
        Pdd<3> pdd(pp,f,q);
        est = pdd.pdd(nof_nodes,node_coord);
    }

    return est;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);

    if (argc != 6) {
        printf("Bad arguments (%d)  -  exit.\n",argc);
        return -1;
    }

#if 1
    int dim = atoi(argv[1]);
    int nof_nodes = atoi(argv[2]);
    char * shm_input_handler = argv[3];
    char * shm_output_handler = argv[4];
    char * shm_config_handler = argv[5];

    int input_size = sizeof(double)*nof_nodes*dim;
    int output_size = sizeof(double)*nof_nodes;
    int config_size = sizeof(struct mc_mpi_config);

    int shm_input_fd = shm_open(shm_input_handler, O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_input_fd < 0) {
        perror("shm_open() 3");
    }

    int shm_output_fd = shm_open(shm_output_handler, O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_output_fd < 0) {
        perror("shm_open() 4");
    }

    int shm_config_fd = shm_open(shm_config_handler, O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_config_fd < 0) {
        perror("shm_open()");
    }

    //create shared memory for node_coord (input)
    void *shm_node_coord = mmap(NULL,input_size,PROT_READ | PROT_WRITE,MAP_SHARED,shm_input_fd,0);
    if (shm_node_coord == MAP_FAILED) {
        perror("mmap()");
    }

    //create shared memory for sol_est (output)
    void *shm_sol_est = mmap(NULL,output_size,PROT_READ | PROT_WRITE,MAP_SHARED,shm_output_fd,0);
    if (shm_sol_est == MAP_FAILED) {
        perror("mmap()");
    }

    void *shm_config = mmap(NULL,config_size,PROT_READ | PROT_WRITE,MAP_SHARED,shm_config_fd,0);
    if (shm_config == MAP_FAILED) {
        perror("mmap()");
    }

    if (close(shm_input_fd) == -1)
        perror("close()");

    if (close(shm_output_fd) == -1)
        perror("close()");

    if (close(shm_config_fd) == -1)
        perror("close()");

    double *node_coord = (double *)shm_node_coord;
    double *sol_est = (double *)shm_sol_est;
    struct mc_mpi_config *common_data = (struct mc_mpi_config *)shm_config;
#endif

    //////////////////////////////////////////////////
    //              MPI code
    //////////////////////////////////////////////////

    int rank;
    int nof_mpi_processes;
    MPI_Comm_size(MPI_COMM_WORLD,&nof_mpi_processes);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    printf("rank = %d of (%d)\n",rank,nof_mpi_processes);

#if 1
    int local_nof_nodes = nof_nodes/nof_mpi_processes;
    enum impl_t impl = IMPL_THREADS;  //IMPL_CL

#if 1
    std::shared_ptr<const dolfin::Expression> f(&common_data->f);
    std::shared_ptr<const dolfin::Expression> q(&common_data->q);
#else
    std::shared_ptr<const dolfin::Expression> &f(common_data->f);
    std::shared_ptr<const dolfin::Expression> &q(common_data->q);
#endif

    std::vector<double> local_sol_est(local_nof_nodes);
    local_sol_est = montecarlo(common_data->D,dim,
                               node_coord + local_nof_nodes*dim*rank, local_nof_nodes,
                               f,q,
                               common_data->walks, common_data->btol,common_data->threads,impl);

    for (int i=0; i<local_nof_nodes; ++i) {
        sol_est[local_nof_nodes*rank + i] = local_sol_est[i];
    }
#endif

    //release shared memory and handler

#if 1
    if (munmap(shm_node_coord,input_size) == -1) {
        perror("munmap()");
    }

    if (munmap(shm_sol_est,output_size) == -1) {
        perror("munmap()");
    }

    if (munmap(shm_config,config_size) == -1) {
        perror("munmap()");
    }
#endif

#if 0
    if (shm_unlink(shm_input_handler)) {
        perror("shm_unlink()");
    }

    if (shm_unlink(shm_output_handler)) {
        perror("shm_unlink()");
    }

    if (shm_unlink(shm_config_handler)) {
        perror("shm_unlink()");
    }
#endif

    MPI_Finalize();

    return 0;
}
