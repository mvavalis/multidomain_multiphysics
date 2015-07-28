#ifndef __HYBRIDMC_H__
#define __HYBRIDMC_H__

#include <vector>
#include <string>

extern const bool opencl;

#ifdef OPENCL_SUPPORT

// MonteCarlo entry point
//
// the threads argument is ignored, just provide a defauld value to keep the API
// similar to the cpu version
//
std::vector<double> montecarlo(double *D, int dim, double* node_coord, int nof_nodes,
                               const std::string &f, const std::string &q,
                               const int walks, const double btol, const int threads = 6, const int mpi_workers = 0);
#endif

#include <memory>

namespace dolfin {
    class Expression;
}

std::vector<double> montecarlo(double *D, int dim, double* node_coord, int nof_nodes,
                               std::shared_ptr<dolfin::Expression> f, std::shared_ptr<dolfin::Expression> q,
                                   const int walks, const double btol, const int threads, const int mpi_workers = 0);

#endif
