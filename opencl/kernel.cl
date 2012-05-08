__kernel void gausslegendreKernel(__global float *Anew,__global float *Bnew, __global const float *A, __global const float *B, __global const float *angulardataA, __global const float *angulardataB, __constant const float *xmap, __constant const float *wmap, const float m0, const float Ddomegasquared, const int N){
	const uint tid = get_global_id(0);

	float Asum=1.0;
	float Bsum=m0;
	float Asquared;
	float Bsquared;

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
}

float gausschebyA(__local const float *args, __constant const float *angx, __constant const float *angw, const int N){
	float sum=0;
	for(int n=0;n<N;n++){
		sum += angw[n]*(2.0/M_PI * (-2.0/3.0*args[1] + (1 + args[1]/args[0])*sqrt(args[0]*args[1])*angx[n] - 4.0/3.0*args[1]*angx[n]*angx[n]) *exp(-(args[0]+args[1]-2*sqrt(args[0]*args[1])*angx[n])/(args[3]*args[3])));
	}

	return sum;
}

float gausschebyB(__local const float *args, __constant const float *angx, __constant const float *angw, const int N){
	float sum=0;
	for(int n=0;n<N;n++){
		sum += angw[n]*(2.0/M_PI * (args[0] + args[1] - 2.0*sqrt(args[0]*args[1])*angx[n])
		*exp(-(args[0]+args[1]-2.0*sqrt(args[0]*args[1])*angx[n])/(args[3]*args[3])));
	}

	return sum;
}

__kernel void angularKernel(__global float *angulardataA, __global float *angulardataB, __global const float *xmap, __constant const float *angx, __constant const float *angw, const int xi, const float omega, const int N, const int Nang){
	const uint tid = get_global_id(0);

	__local float args[4];

	args[1]=xmap[tid];
	args[2]=0;
	args[3]=omega;
	for(int xii=0;xii<N;xii++){
		args[0]=xmap[xii];
		angulardataA[xii + tid*N]=gausschebyA(args, angx, angw, Nang);
		angulardataB[xii + tid*N]=gausschebyB(args, angx, angw, Nang);
	}
}
