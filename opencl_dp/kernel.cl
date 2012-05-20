// #if CONFIG_USE_DOUBLE
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

// real_t
typedef double real_t;
typedef double2 real2_t;
#define FFT_PI 3.14159265358979323846
#define FFT_SQRT_1_2 0.70710678118654752440

//#else

// real_t
//typedef float real_t;
//typedef float2 real2_t;
//#define FFT_PI       3.14159265359f
//#define FFT_SQRT_1_2 0.707106781187f

//#endif

__kernel void gausslegendreKernel(__global real_t *Anew,__global real_t *Bnew, __global const real_t *A, __global const real_t *B, __global const real_t *angulardataA, __global const real_t *angulardataB, __constant const real_t *xmap, __constant const real_t *wmap, const real_t m0, const real_t Ddomegasquared, const int N, __global real_t *epsA, __global real_t *epsB){
	const uint tid = get_global_id(0);

	real_t Asum=1.0;
	real_t Bsum=m0;
	real_t Asquared;
	real_t Bsquared;

	for(int yi=0;yi<N;yi++){
		Asquared=pow(A[yi],2);
		Bsquared=pow(B[yi],2);
		Asum+=wmap[yi]*Ddomegasquared*(xmap[yi]*A[yi]/(xmap[yi]*Asquared+Bsquared)
				*angulardataA[tid + yi*N]);
		Bsum+=wmap[yi]*Ddomegasquared*(xmap[yi]*B[yi]/(xmap[yi]*Asquared+Bsquared)
				*angulardataB[tid + yi*N]);
	}
	Anew[tid]=Asum;
	Bnew[tid]=Bsum;

	epsA[tid]=sqrt(pow(A[tid]-Asum,2));
	epsB[tid]=sqrt(pow(B[tid]-Bsum,2));
}

real_t gausschebyA(__local const real_t *args, __constant const real_t *angx, __constant const real_t *angw, const int N){
	real_t sum=0;
	for(int n=0;n<N;n++){
		sum += angw[n]*(2.0/M_PI * (-2.0/3.0*args[1] + (1 + args[1]/args[0])*sqrt(args[0]*args[1])*angx[n] - 4.0/3.0*args[1]*angx[n]*angx[n]) *exp(-(args[0]+args[1]-2*sqrt(args[0]*args[1])*angx[n])/(args[3]*args[3])));
	}

	return sum;
}

real_t gausschebyB(__local const real_t *args, __constant const real_t *angx, __constant const real_t *angw, const int N){
	real_t sum=0;
	for(int n=0;n<N;n++){
		sum += angw[n]*(2.0/M_PI * (args[0] + args[1] - 2.0*sqrt(args[0]*args[1])*angx[n])
		*exp(-(args[0]+args[1]-2.0*sqrt(args[0]*args[1])*angx[n])/(args[3]*args[3])));
	}

	return sum;
}

__kernel void angularKernel(__global real_t *angulardataA, __global real_t *angulardataB, __global const real_t *xmap, __constant const real_t *angx, __constant const real_t *angw, const int xi, const real_t omega, const int N, const int Nang){
	const uint tid = get_global_id(0);

	__local real_t args[4];

	args[1]=xmap[tid];
	args[2]=0;
	args[3]=omega;
	for(int xii=0;xii<N;xii++){
		args[0]=xmap[xii];
		angulardataA[xii + tid*N]=gausschebyA(args, angx, angw, Nang);
		angulardataB[xii + tid*N]=gausschebyB(args, angx, angw, Nang);
	}
}
