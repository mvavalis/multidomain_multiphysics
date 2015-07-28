//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
//#pragma OPENCL EXTENSION cl_khr_fp64: enable

//inline double CPPCODE_F(double *x) { return x[0]*x[1]*(x[0]-1.)*(x[1]-1.); }
//inline double CPPCODE_Q(double *x) { return - 2*(x[0]*(x[0]-1.) + x[1]*(x[1]-1.)); }

#define AGGRESSIVE_OPTIMIZATIONS
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

void update_y_2D(double x[], double y[], double d, unsigned int *seed)
{
	double angle = rand_uniform(seed)*(2.*M_PI);
	double rad;
	//rad = rand_uniform(seed)*d*0.9;
#ifdef AGGRESSIVE_OPTIMIZATIONS
	do {rad = rand_uniform(seed)*d;} while (((4.*rad)/(d*d))*native_log( (convert_float(d/rad)) ) < rand_uniform(seed)*(4./(M_E*d)));
	y[0] = x[0] + rad*native_cos(convert_float(angle));
	y[1] = x[1] + rad*native_sin(convert_float(angle));
#else
	do {rad = rand_uniform(seed)*d;} while (((4.*rad)/(d*d))*log(d/rad) < rand_uniform(seed)*(4./(M_E*d)));
	y[0] = x[0] + rad*cos(angle);
	y[1] = x[1] + rad*sin(angle);
#endif
}

void update_x_2D(double x[], double d, unsigned int *seed)
{
	double angle = rand_uniform(seed)*(2.*M_PI);
#ifdef AGGRESSIVE_OPTIMIZATIONS
	x[0] += d*native_cos(convert_float(angle));
	x[1] += d*native_sin(convert_float(angle));
#else
	x[0] += d*cos(angle);
	x[1] += d*sin(angle);
#endif
}

double estimate_point_2D(double x[])
{
    return CPPCODE_F(x);
}

double estimate_walk_2D(double x[], double d)
{
    return ((d*d)/4.)*CPPCODE_Q(x);
}

double calc_sphere_rad2D(double x[], double D[])
{
	double min[4];
	unsigned int A, B, C;
	unsigned int index;

	min[0] = D[0] - x[0];
	min[1] = x[0];
	min[2] = D[1] - x[1];
	min[3] = x[1];
#if 0
	A = (min[1]<min[0]);
	B = (min[2]<min[0]) & (min[2]<min[1]);
	C = (min[3]<min[0]) & (min[3]<min[1]) & (min[3]<min[2]);
	index = ((C|B)<<1) | (C|(A&(!B)))&1;
#else
	A = (min[1] < min[0]);
	B = (min[3] < min[2]);
	C = (min[B+2] < min[A]);
	index = (C<<1) | ((B&C)|(A&(!C))&1);
#endif
	return min[index];
}

void random_walk_2D(unsigned int *seed, double x[],
		double y[], double D[], double btol, double *est)
{
	double d;
	double t;

	t = 0;
	while ( (d = calc_sphere_rad2D(x, D)) > btol ) {
		update_y_2D(x, y, d, seed);
		t += estimate_walk_2D(y, d);
		update_x_2D(x, d, seed);
	}
	t += estimate_point_2D(x);
	*est = t;
}

__kernel void DoRandomWalks2D(__global const double D[],
		__global const double x[],
		__global double estimation[],
		unsigned int num_walks,
		double btol,
		unsigned int nodes)
{
	double d;
	local double est[1024];
	local char terminate;
	private double _D[2];
	private double _x[2], _y[2];
	int i;
	private unsigned int seed = get_global_id(0)%MAXRAND + 1;
	// Did not use ints to store get_group_id and get_local_id in order to
	// gain 256 more threads for walkers ...
	// this boosts performance as well as the accuracy of the resuls

	_D[0] = D[0];
	_D[1] = D[1];
	_x[0] = x[get_group_id(0)*2];
	_x[1] = x[get_group_id(0)*2+1];
	// Check whether walks need to be walked or not
	if ( get_group_id(0) >= nodes ) {
		return;
	}
	if ( get_local_id(0) == 0 ) {
		terminate = 0 ;
		d = calc_sphere_rad2D(_x, _D);
		if ( d < 0 ) {
			estimation[get_group_id(0) ] = 0;
			terminate = 1;
		}	else if ( d < btol ) {
			terminate = 2;
			estimation[get_group_id(0)] = estimate_point_2D(_x);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (terminate ) {
		return;
	}
	// Walk the walks
	est[get_local_id(0) ] = 0;
	for ( i=0; i<num_walks; ++i ) {
		_x[0] = x[get_group_id(0)*2];
		_x[1] = x[get_group_id(0)*2+1];
		random_walk_2D(&seed, _x, _y, _D, btol, &d);
		est[get_local_id(0)] += d/(double)(num_walks*get_local_size(0));
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// Sum the walks
	if ( get_local_id(0) == 0 ) {
		d = est[0];
		for ( i=get_local_size(0); i>0; --i ) {
			d+= est[i];
		}
		estimation[get_group_id(0)] = d;
	}
}
