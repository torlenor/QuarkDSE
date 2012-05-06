__kernel void gausslegendreKernel(__global float *Anew,__global float *Bnew, __global const float *A, __global const float *B, __global const float *angulardataA, __global const float *angulardataB, __global const float *xmap, __global const float *wmap, const float m0, const float omega, const float D, const int N){
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
