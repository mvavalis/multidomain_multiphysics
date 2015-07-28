#ifndef MONTECARLO_CL_H
#define MONTECARLO_CL_H
#include "dd.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>

template<int dim>
class MonteCarloCL
{
public:
    MonteCarloCL(int nof_nodes, double *node_coord, const pdd_prm<dim> &pp)
        :nof_nodes(nof_nodes), node_coord(node_coord), pp(pp) { };

    bool init();

    std::vector<double> execute();
    void clear();
    size_t get_workgroup() { return max_workgroup; };

private:

    void set_workgroup(size_t num) { max_workgroup = num; } ;
    size_t max_workgroup;
    std::string load_program(const std::string file);

    int nof_nodes;
    double *node_coord;
    const pdd_prm<dim> pp;
};

#include "MonteCarloCL_impl.h"

#endif
