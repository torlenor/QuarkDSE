__kernel void gausslegendreKernel(__global float *Anew,__global float *Bnew, __global const float *A, __global const float *B, __global const float *angulardataA, __global const float *angulardataB, __constant const float *xmap, __constant const float *wmap, const float m0, const float omega, const float D, const int N){
	uint tid = get_global_id(0);

	Anew[tid]=1.0;
	Bnew[tid]=m0;
	for(int yi=0;yi<N;yi++){
		Anew[tid]+=wmap[yi]*D/(omega*omega)*(xmap[yi]*A[yi]/(xmap[yi]*A[yi]*A[yi]+B[yi]*B[yi])
				*angulardataA[tid + yi*N]);
		Bnew[tid]+=wmap[yi]*D/(omega*omega)*(xmap[yi]*B[yi]/(xmap[yi]*A[yi]*A[yi]+B[yi]*B[yi])
				*angulardataB[tid + yi*N]);
	}
}

float gausschebyA(const float *args, __constant const float *angx, __constant const float *angw, const const int N){
	float sum=0;
	for(int n=0;n<N;n++){
		sum += angw[n]*(2.0/M_PI * (-2.0/3.0*args[1] + (1 + args[1]/args[0])*sqrt(args[0]*args[1])*angx[n] - 4.0/3.0*args[1]*angx[n]*angx[n]) *exp(-(args[0]+args[1]-2*sqrt(args[0]*args[1])*angx[n])/(args[3]*args[3])));
	}

	return sum;
}

float gausschebyB(const float *args, __constant const float *angx, __constant const float *angw, const const int N){
	float sum=0;
	for(int n=0;n<N;n++){
		sum += angw[n]*(2.0/M_PI * (args[0] + args[1] - 2.0*sqrt(args[0]*args[1])*angx[n])
		*exp(-(args[0]+args[1]-2.0*sqrt(args[0]*args[1])*angx[n])/(args[3]*args[3])));
	}

	return sum;
}

__kernel void angularKernel(__global float *angulardataA, __global float *angulardataB, __global const float *xmap, __constant const float *angx, __constant const float *angw, const int xi, const float omega, const int N, const int Nang){
	uint tid = get_global_id(0);

	float args[4];

	args[0]=xmap[xi];
	args[1]=xmap[tid];
	args[2]=0;
	args[3]=omega;
	angulardataA[xi + tid*N]=gausschebyA(args, angx, angw, Nang);
	angulardataB[xi + tid*N]=gausschebyB(args, angx, angw, Nang);
}
