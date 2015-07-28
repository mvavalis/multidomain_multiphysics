#ifndef _PDD_H_
#define _PDD_H_

#include <memory>
#include <iostream>
#include <vector>
#include <ctime>
#include <cassert>
#include "dd.h"

#include "Problem_dolfin.h"
#include "MCDriver.h"
#include "more/euclid_norm.h"

template <int dim>
class Pdd
{
private:
    const struct pdd_prm<dim> &pp;
    Problem<dim> Prob;
public:
    Pdd(const struct pdd_prm<dim> &pp,
        std::shared_ptr<const dolfin::Expression> f,
        std::shared_ptr<const dolfin::Expression> q) :
        pp(pp), Prob(pp.D,f,q) {}

#ifdef OPENCL_SUPPORT
    Pdd(const struct pdd_prm<dim> &pp) :
        pp(pp), Prob(pp.D) {}
#endif

    std::vector<double> pdd(int nof_nodes, double* node_coord);
};

//========================== 2D or 3D ==========================
#define print_mc_est(dim) \
do {std::cout<<"sol_est: "<<std::endl; \
    for (int i=0; i<nof_nodes; i++) { \
        for (int j=0; j<(dim); j++) {std::cout<<node_coord[i][j]<<" \t";} \
        std::cout<<": "<<mc_sol_est[i]<< \
                   " (test_u: "<<Prob.test_u(node_coord[i])<< \
                   ", diff:"<<(Prob.test_u(node_coord[i])-mc_sol_est[i])<<")"<< \
                   std::endl; \
    } \
} while (0)

#define print_elapsed_time() \
do {std::cout<<std::endl<<"time elapsed: "<<(time(NULL) - start_tm)<<"s"<<std::endl;} while (0)

//========================== 2D or 3D ==========================

template <int dim>
std::vector<double> Pdd<dim>::pdd(int nof_nodes, double* node_coord)
{

  time_t start_tm = time(NULL);

  // ==MONTE CARLO==
  std::vector<double> mc_sol_est(nof_nodes); //monte carlo solution estimates
  MCDriver<dim> mc(Prob, pp);
  if (nof_nodes != 0) {
    mc_sol_est = mc.monte_carlo(nof_nodes, node_coord);
    print_elapsed_time();
    #define PDD_TEST 0
    #if PDD_TEST == 1
    print_mc_est(dim);
    //calc_norm<dim>(node_coord, mc_sol_est, pp.D, nof_nodes);
    #endif
  }

  return mc_sol_est;
}

#endif
