#ifndef MC_MPI_DATA_H_
#define MC_MPI_DATA_H_

#include <memory>
#include <dolfin/function/Expression.h>

struct mc_mpi_config {
#if 1
    dolfin::Expression f;
    dolfin::Expression q;
#else
    std::shared_ptr<const dolfin::Expression> f;
    std::shared_ptr<const dolfin::Expression> q;
#endif
    double D[3];  //max number of supported dimensions
    int walks;
    double btol;
    int threads;
};

#endif
