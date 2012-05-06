#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <sstream>

#include <gsl/gsl_sf_legendre.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "kernel.h"

#include "cpufuncts.hpp"
#include "gpufuncts.hpp"

using namespace std;

void mapping(float *xmap, float *wmap, float *x, float *w, float a, float b, float s, int N){
	float g=log(1.0 + (b-a)/s);
	for(int i=0;i<N;i++){
		xmap[i]=a + s*(exp(g*x[i]) - 1.0)/(1.0 + exp(1) - exp(x[i]));
		wmap[i]=w[i]*(s*g*exp(g*x[i]) + (xmap[i] - a)*exp(x[i]))/(1.0+exp(1.0)-exp(x[i]));
	}
}

float angularA(float *args){
	float x=args[0];
	float y=args[1];
	float z=args[2];
	float w=args[3];

	return 2.0/M_PI * (-2.0/3.0*y + (1 + y/x)*sqrt(x*y)*z - 4.0/3.0*y*z*z)
		*exp(-(x+y-2*sqrt(x*y)*z)/(w*w));
}
float angularB(float *args){
	float x=args[0];
	float y=args[1];
	float z=args[2];
	float w=args[3];

	return 2.0/M_PI * (x + y - 2.0*sqrt(x*y)*z)
		*exp(-(x+y-2.0*sqrt(x*y)*z)/(w*w));
}

int main(int argc, char *argv[]){
	// Parameters
	float a=1E-4, b=5E4; // IR/UV cutoff
	float D=16; // GeV^-2
	float omega=0.5; // GeV
	float A0=1, B0=0.4; // Initial values for A(x),B(x)
	float m0=0;

	// Parameter from command line
	if(argc<2){
		m0=0;
	}else{
		m0=atof(argv[1]);
	}

	int iter=16; // How many iterations
	int N=pow(2,10); // Number of discretized values for integration
	int Nang=pow(2,7); // Number of discretized values for integration
	float s=1; // Mapping parameter

	cl::Buffer b_Anew, b_Bnew, b_A, b_B, b_xmap, b_wmap, b_angulardataA, b_angulardataB;

	// OpenCL initialization
	// select platform and device
	if (argc != 4) {
		SetupDevice(0, 0);
	} else {
		SetupDevice(atoi(argv[2]), atoi(argv[3]));
	}

	// load kernel string
	LoadOpenCLKernel(kernel_str);
	// select a main kernel function
	ker = cl::Kernel(prog, "gausslegendreKernel");
	// allocate OpenCL buffers
	b_Anew = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_float));
	b_Bnew = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_float));
	b_A = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_float));
	b_B = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_float));
	b_xmap = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_float));
	b_wmap = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_float));
	b_angulardataA = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*N*sizeof(cl_float));
	b_angulardataB = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*N*sizeof(cl_float));

	// map arguments

	// Main part of quark DSE
	// Write parameters to stdout
	cout << endl << "Quark DSE Solver on GPU/OpenCL v0.0" << endl;
	cout << "(c) Hans-Peter Schadler" << endl << endl;
	cout << "Physical parameter" << endl;
	cout << "IR/UV cutoff: a=" << a << " b=" << b << endl;
	cout << "D=" << D << endl;
	cout << "Omega=" << omega << endl;
	cout << "m0=" << m0 << endl << endl;
	cout << "Numerical parameter" << endl;
	cout << "Initial values: A0=" << A0 << " B0=" << B0 << endl;
	cout << "Number of iterations: " << iter << endl;
	cout << "Number of integration points: " << N << endl;
	cout << "Number of angular integration points: " << Nang << endl << endl;

	cout << "Starting calculation... " << endl << endl;

	// Integraten nodes and weights
	float *xmap, *w, *wmap, *x, *dtmpa;
	x = new float[N];
	xmap = new float[N];
	w = new float[N];
	wmap = new float[N];
	dtmpa = new float[N];
	
	// Working variables
	float *A, *Anew;
	float *B, *Bnew;
	A = new float[N];
	B = new float[N];
	Anew = new float[N];
	Bnew = new float[N];
	float *angulardataA = new float[N*N];
	float *angulardataB = new float[N*N];

	// Initialize initial arrays
	for(int i=0;i<N;i++){
		A[i]=A0;
		B[i]=B0;
	}

	memcpy(Anew,A,sizeof(float)*N);
	memcpy(Bnew,B,sizeof(float)*N);

	// Calculate weights, nodes and remap
	cout << "\tCalculating weights, notes and remap on CPU... " << flush;
	gauleg(0,1,x,w,N);
	mapping(xmap, wmap, x, w, a, b, s, N);
	cout << "done!" << endl;

	q.enqueueWriteBuffer(b_xmap, CL_TRUE, 0, N*sizeof(cl_float), xmap);
	q.enqueueWriteBuffer(b_wmap, CL_TRUE, 0, N*sizeof(cl_float), wmap);	

	float args[4];

	cout << "\tSetting kernel args... " << flush;
	// execute the Gauss-Legendre Integration for A(x), B(x)
	ker.setArg(0, b_Anew);
	ker.setArg(1, b_Bnew);
	ker.setArg(2, b_A);
	ker.setArg(3, b_B);
	ker.setArg(4, b_angulardataA);
	ker.setArg(5, b_angulardataB);
	ker.setArg(6, b_xmap);
	ker.setArg(7, b_wmap);
	ker.setArg(8, m0);
	ker.setArg(9, omega);
	ker.setArg(10, D);
	ker.setArg(11, N);
	cout << "done!" << endl;

	for(int i=0;i<iter;i++){
		cout << "\t--- Iteration " << i << " ---" << endl;
		cout << "\tAngular integration on CPU... " << flush;
		for(int xi=0;xi<N;xi++)
			for(int yi=0;yi<N;yi++){
				args[0]=xmap[xi];
				args[1]=xmap[yi];
				args[2]=0;
				args[3]=omega;
				angulardataA[xi + yi*N]=gausscheby(angularA, args, 2, Nang);
				angulardataB[xi + yi*N]=gausscheby(angularB, args, 2, Nang);
			}
		cout << "done!" << endl;

		cout << "\tWriting buffers to GPU... " << flush;
		q.enqueueWriteBuffer(b_A, CL_TRUE, 0, N*sizeof(cl_float), A);
		q.enqueueWriteBuffer(b_B, CL_TRUE, 0, N*sizeof(cl_float), B);
		q.enqueueWriteBuffer(b_angulardataA, CL_TRUE, 0, N*N*sizeof(cl_float), angulardataA);
		q.enqueueWriteBuffer(b_angulardataB, CL_TRUE, 0, N*N*sizeof(cl_float), angulardataB);
		cout << "done!" << endl;

		cout << "\tFiring up kernels on GPU... " << flush;
		cl::Event event;
		q.enqueueNDRangeKernel(ker, cl::NullRange, cl::NDRange(N), cl::NDRange(128), NULL, &event);
		cout << "done!" << endl;
		
		cout << "\tReading buffers from GPU... " << flush;
		// read output data
		q.enqueueReadBuffer(b_Anew, CL_TRUE, 0, N*sizeof(cl_float), Anew);
		q.enqueueReadBuffer(b_Bnew, CL_TRUE, 0, N*sizeof(cl_float), Bnew);
		cout << "done!" << endl;

		memcpy(A,Anew,sizeof(float)*N);
		memcpy(B,Bnew,sizeof(float)*N);
	}

	ofstream fout;
	fout.open("dressing.data");

	for(int xi=0;xi<N;xi++){
		fout << xmap[xi] << " " << A[xi] << " " << B[xi] << " " << B[xi]/A[xi] << endl;
	}
	fout.close();

	cout << endl << endl << "done" << endl;

	return 0;
}

