//compile with -std=c++11

#include <iostream>
#include <vector>

#ifndef OPENCL_SUPPORT
#define OPENCL_SUPPORT
#endif
#include <hybridmc/mc.h>

int main(int argc, char *argv[]) {
    std::cout << "\n\n\n****    TEST    ****\n\n\n";

    // 2D boundary
    std::vector<double> dim(2);
    dim[0] = .1;  //x-dimension size
    dim[1] = .1;  //y-dimension size

    const int num_of_points = 128;
    const int serialized_coord_size = num_of_points * dim.size();

    // serialized coordinates of points
    std::vector<double> data(serialized_coord_size);

    //random coordinates
    for (int point=0; point<num_of_points; ++point) {
        for (unsigned int coord=0; coord<dim.size(); ++coord) {
            data[point*dim.size() + coord] = dim[coord]/(point + 1);
        }
    }

    //PDE definition
    std::string f_ref = "(x[0]-1)*(x[0])*(x[1]-1)*(x[1])";
    std::string q_ref = "-1 * (2 * (-1 + (x[0])) * (x[0]) + 2 * (x[1]) * (-1 + (x[1])))";

    const int num_of_walks = 5000;
    const double btol = 1e-13;

    //run the OpenCL version
    std::vector<double> result = montecarlo(dim.data(),dim.size(),data.data(),data.size()/dim.size()
                                            ,f_ref,q_ref
                                            ,num_of_walks,btol);
    for (std::vector<double>::const_iterator i=result.begin(); i!=result.end(); ++i)
      std::cout << "result: " << *i << std::endl;

    return 0;
}
