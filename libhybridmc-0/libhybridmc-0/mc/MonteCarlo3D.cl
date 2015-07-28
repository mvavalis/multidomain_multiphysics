//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
//#pragma OPENCL EXTENSION cl_khr_fp64: enable

//inline double CPPCODE_F(double *x) { return exp(M_SQRT2*M_PI*(x[0]-1.)) * precise_sin(M_PI*(x[1]-1. + x[2]-1.)) + (1./6.) * ((x[0]-1.)*(x[0]-1.)*(x[0]-1.)  +  (x[1]-1.)*(x[1]-1.)*(x[1]-1.)  +  (x[2]-1.)*(x[2]-1.)*(x[2]-1.)); }
//inline double CPPCODE_Q(double *x) { return - (x[0]-1. + x[1]-1. + x[2]-1.); }

#define AGGRESSIVE_OPTIMIZATIONS
#define MAXRAND 65536

#ifdef AGGRESSIVE_OPTIMIZATIONS
#define sin(A) native_sin(convert_float((A)))
#define cos(A) native_cos(convert_float((A)))
#define log(A) native_log(convert_float((A)))
//#define exp(A) convert_double(native_exp(convert_float((A))))
#endif
#define precise_sin sin

int rand(unsigned int *seed)
{
	*seed = *seed * 1103515245 + 12345;
	return (*seed % ((unsigned int)MAXRAND + 1));
}
double rand_uniform(unsigned int *seed) {
	double r = rand(seed)/(double)MAXRAND;
	return r;
}

void update_y_3D(double x[], double y[], double d, unsigned int *seed)
{
	double theta = rand_uniform(seed)*(2.*M_PI);
	double rad, phi;
	do {rad = rand_uniform(seed)*d;  phi = rand_uniform(seed)*M_PI;}
	while ((3./(d*d*d))*rad*(d-rad)*sin(phi) < rand_uniform(seed)*((3./4.)*d));
	y[0] = x[0] + rad*sin(phi)*cos(theta);
	y[1] = x[1] + rad*sin(phi)*sin(theta);
	y[2] = x[2] + rad*cos(phi);
}

void update_x_3D(double x[], double d, unsigned int *seed)
{
	double theta = rand_uniform(seed)*(2.*M_PI);
	double phi = rand_uniform(seed)*(M_PI);
	double sin_phi = sin(phi);
	x[0] += d*sin_phi*cos(theta);
	x[1] += d*sin_phi*sin(theta);
	x[2] += d*cos(phi);
}

double estimate_point_3D(double x[])
{
    return CPPCODE_F(x);
}

double estimate_walk_3D(double x[], double d)
{
    return (d*d/6.) * CPPCODE_Q(x);
}

double calc_sphere_rad3D(double x[], double D[])
{
	double min = (D[0]-x[0]);
	if (x[0] < min) {min = x[0];}
	if (D[1]-x[1] < min) {min = D[1]-x[1];}
	if (x[1] < min) {min = x[1];}
	if (D[2]-x[2] < min) {min = D[2]-x[2];}
	if (x[2] < min) {min = x[2];}
	return min;
}

void random_walk_3D(unsigned int *seed, double x[],
		double y[], double D[], double btol, double *est)
{
	double d;
	double t;
	int i = 0;

	t = 0.;
	while ( (d = calc_sphere_rad3D(x, D)) > btol ) {
		update_y_3D(x, y, d, seed);
		t += estimate_walk_3D(y, d);
		update_x_3D(x, d, seed);
	}
	t += estimate_point_3D(x);
	*est = t;
}

__kernel void DoRandomWalks3D(__global const double D[],
		__global const double x[],
		__global double estimation[],
		unsigned int num_walks,
		double btol,
		unsigned int nodes)
{
	double d;
	double mine;
	local double est[1024];
	local char terminate;
	private double _D[3];
	private double _x[3], _y[3];
	unsigned int i;
	private unsigned int seed = get_global_id(0)%MAXRAND + 1;
	const unsigned int me = get_local_id(0);
	const unsigned int group = get_group_id(0);
	_D[0] = D[0];
	_D[1] = D[1];
	_D[2] = D[2];
	est[me] = 0.;
	// Check whether walks need to be walked or not
	if ( me == 0 ) {
		_x[0] = x[group*3];
		_x[1] = x[group*3+1];
		_x[2] = x[group*3+2];
		terminate = 0 ;
		d = calc_sphere_rad3D(_x, _D);
		if ( d < 0 ) {
			estimation[group ] = 0;
			terminate = 1;
		}	else if ( d < btol ) {
			terminate = 2;
			estimation[group] = estimate_point_3D(_x);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (terminate ) {
		return;
	}
	// Walk the walks
	mine = 0.;
	for ( i=0; i<num_walks; ++i ) {
		_x[0] = x[group*3];
		_x[1] = x[group*3+1];
		_x[2] = x[group*3+2];
		random_walk_3D(&seed, _x, _y, _D, btol, &d);
		mine += d/(double)(num_walks*get_local_size(0));
	}
	est[me] = mine;
	barrier(CLK_LOCAL_MEM_FENCE);
	// Sum the walks
	if ( me == 0 ) {
		d = est[0];
		for ( i=1; i<get_local_size(0); i++ ) {
			d += est[i];
		}
		estimation[group] = d;
	}
}
