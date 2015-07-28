#ifndef _MC_H_
#define _MC_H_

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <ctime>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "dd.h"
#include "mc_mpi_data.hpp"
#include "more/Vdcbin.h"

#ifdef OPENCL_SUPPORT
#include "MonteCarloCL.h"
#endif

#include "mpi_exec_path.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <unistd.h>
#include <sys/types.h>

template <int dim>
class MCDriver
{
private:
	struct output {
		//double mnof_steps; //mean of the nof_steps required reach the boundary
		double msol_est; //mean of the solution estimate
	};

	Problem<dim> &Prob;
	const struct pdd_prm<dim> &pp;

#ifdef MY_RAND
        unsigned int seed;
#endif

	gsl_rng *rng; //random number generator
	Vdcbin vdcX, vdcY; //van der Corput sequence
	double btol; //boundary tolerance
	double D[dim], btol_D[dim];  //\Omega: dimension length, and dimension length minus boundary tolerance

	double calc_sphere_rad(const double *x);
	double a(double d);
	void rand_update_x(double *x, double d);
	void quasi_update_x(double *x, double d);
	void rand_update_y(const double *x, double *y, double d);
	void quasi_update_y(const double *x, double *y, double d);

	void init_quasi(int size);
	double quasirand_uniform();

	struct output solve(int nof_walks, const double *x_start);
	static void* mcmain(void *arg);
        std::vector<double> monte_carlo_cpu(int nof_nodes, double *node_coord);
        std::vector<double> monte_carlo_mpi(int nof_nodes, double *node_coord);
#ifdef OPENCL_SUPPORT
        std::vector<double> monte_carlo_opencl(int nof_nodes, double *node_coord);
#endif
public:
	MCDriver(Problem<dim> &Prob, const struct pdd_prm<dim> &pp);
        std::vector<double> monte_carlo(int nof_nodes, double* node_coord);
};

template <int dim>
struct mcjob {
	MCDriver<dim> *mc;
	int nof_walks;
	int nof_nodes;
	double *node_coord;
	double *sol_est;
};

//#include "more/Ranq1.h"
//Ranq1 ranq(time(NULL)); //random number generator
//#define rand_uniform() (ranq.doub())
#ifdef   MY_RAND
#define MAXRAND 65536

int rand(unsigned int *seed)
{
       *seed = *seed * 1103515245 + 12345;
       return (*seed % ((unsigned int)MAXRAND + 1));
}
double rand_uniform(unsigned int *seed) {
 double r = rand(seed)/(double)MAXRAND;
 return r;
}
#define rand_uniform() (rand_uniform(&seed))
#else
#define rand_uniform() (gsl_rng_uniform(rng))
#endif

//============================= 2D =============================
template <> inline double MCDriver<2>::calc_sphere_rad(const double *x)
{
	double min;
	min = (D[0]-x[0]);
	if (x[0] < min) {min = x[0];}
	if (D[1]-x[1] < min) {min = D[1]-x[1];}
	if (x[1] < min) {min = x[1];}
	return min;
}

template <> inline double MCDriver<2>::a(double d)
{
	return d*d/4.;
}

template <> inline void MCDriver<2>::rand_update_x(double *x, double d)
{
	double angle = rand_uniform()*(2.*M_PI);
	x[0] += d*cos(angle);
	x[1] += d*sin(angle);
}

template <> inline void MCDriver<2>::quasi_update_x(double *x, double d)
{
	double angle = quasirand_uniform()*(2.*M_PI);
	x[0] += d*cos(angle);
	x[1] += d*sin(angle);
}

// template <> inline void MCDriver<2>::quasi_update_x(double *x, double d)
// {
// 	double angle = vdcX.doub_cz()*(2.*M_PI);
// 	x[0] += d*cos(angle);
// 	x[1] += d*sin(angle);
// }

template <> inline void MCDriver<2>::rand_update_y(const double *x, double *y, double d)
{
	double angle = rand_uniform()*(2.*M_PI);
	double rad;
	do {rad = rand_uniform()*d;} while (((4.*rad)/(d*d))*log(d/rad) < rand_uniform()*(4./(M_E*d)));
	y[0] = x[0] + rad*cos(angle);
	y[1] = x[1] + rad*sin(angle);
}

template <> inline void MCDriver<2>::quasi_update_y(const double *x, double *y, double d)
{
	double angle = quasirand_uniform()*(2.*M_PI);
	double rad;
	do {rad = quasirand_uniform()*d;} while (((4.*rad)/(d*d))*log(d/rad) < quasirand_uniform()*(4./(M_E*d)));
	y[0] = x[0] + rad*cos(angle);
	y[1] = x[1] + rad*sin(angle);
}

// template <> inline void MCDriver<2>::quasi_update_y(const double *x, double *y, double d)
// {
// 	double angle = vdcY.doub_cz()*(2.*M_PI);
// 	double rad;
// 	do {rad = rand_uniform()*d;} while (((4.*rad)/(d*d))*log(d/rad) < rand_uniform()*(4./(M_E*d)));
// 	y[0] = x[0] + rad*cos(angle);
// 	y[1] = x[1] + rad*sin(angle);
// }

//============================= 3D =============================
template <> inline double MCDriver<3>::calc_sphere_rad(const double *x)
{
	double min = (D[0]-x[0]);
	if (x[0] < min) {min = x[0];}
//  		double ppD[3] = {2., 2., 2.};
//  		Problem<3> Probb(ppD);
	if (D[1]-x[1] < min) {min = D[1]-x[1];}
	if (x[1] < min) {min = x[1];}
	if (D[2]-x[2] < min) {min = D[2]-x[2];}
	if (x[2] < min) {min = x[2];}
	return min;
}

template <> inline double MCDriver<3>::a(double d)
{
	return d*d/6.;
}

template <> inline void MCDriver<3>::rand_update_x(double *x, double d)
{
	double theta = rand_uniform()*(2.*M_PI);
	double phi = rand_uniform()*(M_PI);
	x[0] += d*sin(phi)*cos(theta);
	x[1] += d*sin(phi)*sin(theta);
	x[2] += d*cos(phi);
}

template <> inline void MCDriver<3>::quasi_update_x(double *x, double d)
{
	double theta = quasirand_uniform()*(2.*M_PI);
	double phi = quasirand_uniform()*(M_PI);
	x[0] += d*sin(phi)*cos(theta);
	x[1] += d*sin(phi)*sin(theta);
	x[2] += d*cos(phi);
}

// template <> inline void MCDriver<3>::quasi_update_x(double *x, double d)
// {
// 	double theta = vdcX.doub_cz()*(2.*M_PI);
// 	double phi = rand_uniform()*(M_PI);
// 	x[0] += d*sin(phi)*cos(theta);
// 	x[1] += d*sin(phi)*sin(theta);
// 	x[2] += d*cos(phi);
// }

template <> inline void MCDriver<3>::rand_update_y(const double *x, double *y, double d)
{
	double theta = rand_uniform()*(2.*M_PI);
	double rad, phi;
	do {rad = rand_uniform()*d;  phi = rand_uniform()*M_PI;}
	while ((3./(d*d*d))*rad*(d-rad)*sin(phi) < rand_uniform()*((3./4.)*d));
	y[0] = x[0] + rad*sin(phi)*cos(theta);
	y[1] = x[1] + rad*sin(phi)*sin(theta);
	y[2] = x[2] + rad*cos(phi);
}

template <> inline void MCDriver<3>::quasi_update_y(const double *x, double *y, double d)
{
	double theta = quasirand_uniform()*(2.*M_PI);
	double rad, phi;
	do {rad = quasirand_uniform()*d;  phi = quasirand_uniform()*M_PI;}
	while ((3./(d*d*d))*rad*(d-rad)*sin(phi) < quasirand_uniform()*((3./4.)*d));
	y[0] = x[0] + rad*sin(phi)*cos(theta);
	y[1] = x[1] + rad*sin(phi)*sin(theta);
	y[2] = x[2] + rad*cos(phi);
}

// template <> inline void MCDriver<3>::quasi_update_y(const double *x, double *y, double d)
// {
// 	double theta = vdcY.doub_cz()*(2.*M_PI);
// 	double rad, phi;
// 	do {rad = rand_uniform()*d;  phi = rand_uniform()*M_PI;}
// 	while ((3./(d*d*d))*rad*(d-rad)*sin(phi) < rand_uniform()*((3./4.)*d));
// 	y[0] = x[0] + rad*sin(phi)*cos(theta);
// 	y[1] = x[1] + rad*sin(phi)*sin(theta);
// 	y[2] = x[2] + rad*cos(phi);
// }


//========================== 2D or 3D ==========================
template <int dim>
MCDriver<dim>::MCDriver(Problem<dim> &Prob, const struct pdd_prm<dim> &pp) : Prob(Prob), pp(pp), vdcX(Vdcbin()), vdcY(Vdcbin())
{
#ifdef MY_RAND
        seed = 1;
#endif

        gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(rng, time(NULL));

	int i;
	for (i=0; i<dim; i++) {D[i] = Prob.D[i];}
	btol = pp.btol;
	for (i=0; i<dim; i++) {btol_D[i] = D[i]-btol;}
}

#if RAND_MODE == 2
static double *qarray;
static int qindex;
static int qsize;
template <int dim>
void MCDriver<dim>::init_quasi(int size)
{
	int tmp_index;
	double tmp;

	qsize = size;
	qindex = 0;
	qarray = new double[size];

	for (int i=0; i<size; i++) {
		qarray[i] = vdcX.doub_cz();
// 		qarray[i] = rand_uniform(); /*used for debugging*/
	}

// #define NUMMMM 107
// 	for (int i=0; i<size; i++) {
// 		for (int j=0; j<NUMMMM; j++) {
// 			tmp = qarray[(i+j)];
// 			tmp_index = (i+j) + gsl_rng_uniform_int(rng, (i+NUMMMM)-(i+j));
//
// 			qarray[(i+j)] = qarray[tmp_index];
// 			qarray[tmp_index] = tmp;
// 		}
// 		i+=NUMMMM-1;
// 	}

	for (int i=0; i<size; i++) {
		tmp = qarray[i];
		tmp_index = i + gsl_rng_uniform_int(rng, size-i);

		qarray[i] = qarray[tmp_index];
		qarray[tmp_index] = tmp;
	}
}

template <int dim>
double MCDriver<dim>::quasirand_uniform()
{
	if (qindex == qsize) {
		delete [] qarray;
		init_quasi(qsize);
		std::cout<<"init_quasi(qsize);"<<std::endl;
	}

	return qarray[qindex++];
}
#endif

// static int max_nof_steps = 0;
// static int walks_over_60 = 0;
// template <int dim>
// struct MCDriver<dim>::output MCDriver<dim>::solve(int nof_walks, const double *x_start)
// {
// 	int i, j;
// 	struct output out; //mean of the solution estimate and the nof_steps
// 	double sol_est; //temp estimate of the solution
// 	double x[dim], y[dim], d; //boundary point, ball point, sphere's radius
//
// 	d = calc_sphere_rad(x_start);
// 	if (d < 0.) {return out;} //outside \Omega
// 	if (d < btol) { //very close to the boundary
// 		out.msol_est = Prob.f(x_start);
// 		return out;
// 	}
//
// 	//out.mnof_steps = .0;
// 	out.msol_est = .0;
// 	for (i=0; i<nof_walks; i++) {
// 		for (j=0; j<dim; j++) {x[j] = x_start[j];}
//
//
// 		while ((d = calc_sphere_rad(x)) > btol) {
// 			rand_update_y(x, y, d);
// 			sol_est += a(d)*Prob.q(y);
//
// 			rand_update_x(x, d);
// 			//out.mnof_steps++;
// // 			nof_steps++;
// 		}
//
// 		sol_est += Prob.f(x);
//
// // 		if (nof_steps > max_nof_steps) {
// // 			max_nof_steps = nof_steps;
// // 		}
// // 		if (nof_steps == 60) {
// // 			walks_over_60++;
// // 		}
//
// 		out.msol_est += sol_est/nof_walks;
// 	}
//
// 	//out.mnof_steps /= nof_walks;
//
// // 	std::cout<<"max_nof_steps: "<<max_nof_steps<<std::endl;
// // 	std::cout<<"walks_over_60: "<<walks_over_60<<std::endl;
//
// 	return out;
// }

template <int dim>
struct MCDriver<dim>::output MCDriver<dim>::solve(int nof_walks, const double *x_start)
{
	int i, j;
	struct output out; //mean of the solution estimate and the nof_steps
        out.msol_est = 0.0;
	double sol_est; //temp estimate of the solution
	double x[dim], y[dim], d; //boundary point, ball point, sphere's radius

	d = calc_sphere_rad(x_start);
	if (d < 0) {return out;} //outside \Omega
	if (d < btol) { //very close to the boundary
		out.msol_est = Prob.f(x_start);
		return out;
	}

#if RAND_MODE == 2
	init_quasi(5*nof_walks*30);
#endif

	//out.mnof_steps = .0;
	out.msol_est = .0;
	for (i=0; i<nof_walks; i++) {
		for (j=0; j<dim; j++) {x[j] = x_start[j];}

		sol_est = .0;

// 		if ((d = calc_sphere_rad(x)) > btol) {
// 			quasi_update_y(x, y, d);
// 			sol_est += a(d)*Prob.q(y);
//
// 			quasi_update_x(x, d);
// 			//out.mnof_steps++;
// 		}

		while ((d = calc_sphere_rad(x)) > btol) {
			#if RAND_MODE == 1 /*pseudo*/
				rand_update_y(x, y, d);
			#else /*quasi*/
				quasi_update_y(x, y, d);
			#endif
			sol_est += a(d)*Prob.q(y);

			#if RAND_MODE == 1 /*pseudo*/
				rand_update_x(x, d);
			#else /*quasi*/
				quasi_update_x(x, d);
			#endif
			//out.mnof_steps++;
		}

		sol_est += Prob.f(x);

		out.msol_est += sol_est/nof_walks;
	}

#if RAND_MODE == 2
	delete [] qarray;
#endif

#if 0
	std::cout<<"diff: "<<fabs(Prob.test_u(x_start)-out.msol_est)<<std::endl;
#endif

	//out.mnof_steps /= nof_walks;
	return out;
}

#if INTERFACE_MODE == 2
	double D_temp[3];
#endif
template <int dim>
void* MCDriver<dim>::mcmain(void *arg)
{
	int i;
	struct mcjob<dim> mcj = *(struct mcjob<dim> *)arg;
	struct output out;

	for (i=0; i<mcj.nof_nodes; i++) {
#if INTERFACE_MODE == 1
		out = mcj.mc->solve(mcj.nof_walks, &mcj.node_coord[i*dim]);
		mcj.sol_est[i] = out.msol_est;
#elif INTERFACE_MODE == 2
 		Problem<dim> Prob_temp(D_temp);
		mcj.sol_est[i] = Prob_temp.test_u(&mcj.node_coord[i*dim]);
#endif
	}

	/* here mnof_steps could be returned */
	pthread_exit(NULL);
}

template <int dim>
std::vector<double> MCDriver<dim>::monte_carlo(int nof_nodes, double* node_coord)
{
	std::ofstream exp("experiment", std::ios::out|std::ios::binary);
	exp.write((char*)&nof_nodes, sizeof(nof_nodes));
	exp.write((char*)D, sizeof(double)*dim);
	exp.write((char*)node_coord, sizeof(double)*dim*nof_nodes);
	exp.close();

#ifdef OPENCL_SUPPORT
	if ( pp.impl == IMPL_CL )
            return monte_carlo_opencl(nof_nodes, node_coord);
#endif
        if ( pp.impl == IMPL_MPI )
            return monte_carlo_mpi(nof_nodes, node_coord);
        return monte_carlo_cpu(nof_nodes, node_coord);
}

#ifdef OPENCL_SUPPORT
template <int dim>
std::vector<double> MCDriver<dim>::monte_carlo_opencl(int nof_nodes, double *node_coord)
{
	MonteCarloCL<dim> mcl(nof_nodes, node_coord, pp);
	mcl.init();

	std::cout<<"\n*** Monte Carlo ***"<<std::endl
		<<"nof threads:\t"<<nof_nodes*mcl.get_workgroup()<<std::endl
		<<"nof nodes/jobs:\t"<<nof_nodes<<std::endl
		<<"max work group size:\t" << mcl.get_workgroup() << std::endl
		<<"nof walks:\t"<<pp.nof_walks<<std::endl
		<<"boundary tolerance:\t"<<pp.btol<<std::endl;


	return mcl.execute();
}
#endif


template <int dim>
std::vector<double> MCDriver<dim>::monte_carlo_mpi(int nof_nodes, double *node_coord)
{
    //create the private result (sol_est)
    //map the private input (*node_coord) in a shared memory segment

    //call the MPI implementation (pass the shared memory handler)
    //copy the result from the shared memory to the private result (sol_est)

    //unmap shared memory
    //deallocate shared memory

    //private result
    std::vector<double> sol_est(nof_nodes);

    //create shared memory handlers (files)
    std::string shm_input_handler = "mc_mpi_input";
    std::string shm_output_handler = "mc_mpi_output";
    std::string shm_config_handler = "mc_config";

    //clear previous shared memory

    shm_unlink(shm_input_handler.c_str());
    shm_unlink(shm_output_handler.c_str());
    shm_unlink(shm_config_handler.c_str());

    int shm_input_fd = shm_open(shm_input_handler.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_input_fd < 0) {
        perror("shm_open() 1");
    }

    int shm_output_fd = shm_open(shm_output_handler.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_output_fd < 0) {
        perror("shm_open() 2");
    }

    int shm_config_fd = shm_open(shm_config_handler.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_config_fd < 0) {
        perror("shm_open()");
    }

    int input_size = sizeof(double)*nof_nodes*dim;
    int output_size = sizeof(double)*nof_nodes;
    int config_size = sizeof(struct mc_mpi_config);

    if (ftruncate(shm_input_fd,input_size)) {
        perror("ftruncate()");
    }

    if (ftruncate(shm_output_fd,output_size)) {
        perror("ftruncate()");
    }

    if (ftruncate(shm_config_fd,config_size)) {
        perror("ftruncate()");
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

    //create shared memory for sol_est (output)
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

    //prepare the command to run
    std::string executable = MPI_EXEC_BINARY;
    std::stringstream cmd;
    cmd
        << "mpiexec"
        //<< " --debug"
        << " -n " << pp.mpi_workers
        << " " /*space*/ << executable.c_str()
        /* arguments to the mpi executable */
        << " " /*space*/ << dim
        << " " /*space*/ << nof_nodes
        << " " /*space*/ << shm_input_handler.c_str()
        << " " /*space*/ << shm_output_handler.c_str()
        << " " /*space*/ << shm_config_handler.c_str();

    std::string cmd_buffer = cmd.str();

    //copy the private memory of the input to the shared memory
    for (int i=0; i<nof_nodes*dim; ++i)
        ((double*)shm_node_coord)[i] = node_coord[i];

    //copy the config data to shared memory
    struct mc_mpi_config *common = (struct mc_mpi_config *)shm_config;
#if 1
    memcpy(&common->f,&*Prob.f_expr,sizeof(dolfin::Expression));
    memcpy(&common->q,&*Prob.q_expr,sizeof(dolfin::Expression));
#else
    common->f = Prob.f_expr;
    common->q = Prob.q_expr;
#endif
    for (int i=0; i<dim; ++i)
        common->D[i] = Prob.D[i];
    common->walks = pp.nof_walks;
    common->btol = pp.btol;
    common->threads = pp.max_nof_threads;

    //execute the mpi implementation
    std::cout << "DEBUG: execute: " << cmd_buffer.c_str();
    int status = system(cmd_buffer.c_str());
    if (status == -1) {
        perror("system()");
    }

    //copy the shared memory of the output to the private memory
    std::copy((double*)shm_sol_est,((double*)shm_sol_est) + nof_nodes,sol_est.begin());

    //release shared memory and handler

    if (munmap(shm_node_coord,input_size) == -1) {
        perror("munmap()");
    }

    if (munmap(shm_sol_est,output_size) == -1) {
        perror("munmap()");
    }

    if (munmap(shm_config,config_size) == -1) {
        perror("munmap()");
    }

    if (shm_unlink(shm_input_handler.c_str())) {
        perror("shm_unlink()");
    }

    if (shm_unlink(shm_output_handler.c_str())) {
        perror("shm_unlink()");
    }

    if (shm_unlink(shm_config_handler.c_str())) {
        perror("shm_unlink()");
    }

    return sol_est;
}

template <int dim>
std::vector<double> MCDriver<dim>::monte_carlo_cpu(int nof_nodes, double *node_coord)
{
#if INTERFACE_MODE == 2
	for (int tmpi=0; tmpi<dim; tmpi++) {
		D_temp[tmpi] = D[tmpi];
	}
#endif

	int i, node_offset, nof_threads;
	int nof_jpthread, ntjp; //nof jobs per thread, nof threads with plus one job
	pthread_t *pmc;
	struct mcjob<dim> *mcj;

    //sol_est = new std::vector<double>(nof_nodes);
    std::vector<double> sol_est(nof_nodes);

	nof_threads = (pp.max_nof_threads > nof_nodes) ? nof_nodes:pp.max_nof_threads;
	nof_jpthread = nof_nodes/nof_threads;
	ntjp = nof_nodes - nof_jpthread*nof_threads;

	std::cout<<"\n*** Monte Carlo ***"<<std::endl
	         <<"nof threads:\t"<<nof_threads<<std::endl
	         <<"nof nodes/jobs:\t"<<nof_nodes<<std::endl
	         <<"nof walks:\t"<<pp.nof_walks<<std::endl
	         <<"boundary tolerance:\t"<<pp.btol<<std::endl;

	pmc = new pthread_t[nof_threads];
	mcj = new mcjob<dim>[nof_threads];

	for (i=0; i<ntjp; i++) {
		node_offset = i*(nof_jpthread+1);
		mcj[i].mc         = this;
		mcj[i].nof_walks  = pp.nof_walks;
		mcj[i].nof_nodes  = nof_jpthread+1;
		mcj[i].node_coord = &node_coord[node_offset*dim];
		mcj[i].sol_est    = &sol_est[node_offset];
		pthread_create(&pmc[i], NULL, mcmain, &mcj[i]);
	}

	for (i=ntjp; i<nof_threads; i++) {
		node_offset = ntjp*(nof_jpthread+1) + (i-ntjp)*nof_jpthread;
		mcj[i].mc         = this;
		mcj[i].nof_walks  = pp.nof_walks;
		mcj[i].nof_nodes  = nof_jpthread;
		mcj[i].node_coord = &node_coord[node_offset*dim];
		mcj[i].sol_est    = &sol_est[node_offset];
		pthread_create(&pmc[i], NULL, mcmain, &mcj[i]);
	}

	for (i=0; i<nof_threads; i++) {pthread_join(pmc[i], NULL);}
// 	delete [] pmc;  delete [] mcj;


// std::cout << "aaaaaaaaaaaaaa" << nof_nodes << ", " << dim << std::endl;
// for (int i=0; i<nof_nodes*dim; ++i)
//   std::cout << "mc_input: " << node_coord[i] << std::endl;
//
// for (int i=0; i<dim; ++i)
//   std::cout << "mc_dim: " << pp.D[i] << std::endl;
// for (std::vector<double>::const_iterator i=sol_est.begin(); i!=sol_est.end(); ++i)
//   std::cout << "mc_est: " << *i << std::endl;
//
    return sol_est;
}

#endif
