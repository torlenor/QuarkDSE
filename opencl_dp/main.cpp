#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <sstream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "kernel.h"

#include "cpufuncts.hpp"
#include "gpufuncts.hpp"

using namespace std;

void mapping(double *xmap, double *wmap, double *x, double *w, double a, double b, double s, int N){
	double g=log(1.0 + (b-a)/s);
	for(int i=0;i<N;i++){
		xmap[i]=a + s*(exp(g*x[i]) - 1.0)/(1.0 + exp(1) - exp(x[i]));
		wmap[i]=w[i]*(s*g*exp(g*x[i]) + (xmap[i] - a)*exp(x[i]))/(1.0+exp(1.0)-exp(x[i]));
	}
}

int main(int argc, char *argv[]){
	// Parameters
	double a=1E-4, b=1E5; // IR/UV cutoff
	double D=16; // GeV^-2
	double omega=0.5; // GeV
	double A0=1, B0=0.4; // Initial values for A(x),B(x)
	double m0=0;

	// Parameter from command line
	if(argc<2){
		m0=0;
	}else{
		m0=atof(argv[1]);
	}

	int iter=100; // Maximum number of iterations
	int N=pow(2,12); // Number of discretized values for integration
	int Nang=pow(2,11); // Number of discretized values for integration
	double eps=1E-10;
	double s=1; // Mapping parameter
	int wgsize=64; // Workgroup size

	cl::Buffer b_Anew, b_Bnew, b_A, b_B, b_xmap, b_wmap, b_angx, b_angw, b_angulardataA, b_angulardataB, b_epsA, b_epsB;

	// OpenCL initialization
	// select platform and device
	if (argc != 4) {
		SetupDevice(0, 0);
	} else {
		SetupDevice(atoi(argv[2]), atoi(argv[3]));
		if(atoi(argv[3])==0){
			wgsize=64;
		}else{
			wgsize=1; 
		}
	}


	// load kernel string
	LoadOpenCLKernel(kernel_str);
	// select a main kernel function
	ker = cl::Kernel(prog, "gausslegendreKernel");
	angularker = cl::Kernel(prog, "angularKernel");

	// allocate OpenCL buffers
	b_Anew = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_double));
	b_Bnew = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_double));
	b_A = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_double));
	b_B = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_double));
	b_xmap = cl::Buffer(ctx, CL_MEM_READ_ONLY, N*sizeof(cl_double));
	b_wmap = cl::Buffer(ctx, CL_MEM_READ_ONLY, N*sizeof(cl_double));
	b_angx = cl::Buffer(ctx, CL_MEM_READ_ONLY, Nang*sizeof(cl_double));
	b_angw = cl::Buffer(ctx, CL_MEM_READ_ONLY, Nang*sizeof(cl_double));
	b_angulardataA = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*N*sizeof(cl_double));
	b_angulardataB = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*N*sizeof(cl_double));
	b_epsA = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_double));
	b_epsB = cl::Buffer(ctx, CL_MEM_READ_WRITE, N*sizeof(cl_double));

	// Main part of quark DSE
	// Write parameters to stdout
	cout << endl << "Quark DSE Solver on GPU/OpenCL v1.0" << endl;
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
	double *x = new double[N];
	double *xmap = new double[N];
	double *w = new double[N];
	double *wmap = new double[N];
	double *dtmpa = new double[N];
	double *angx = new double[Nang];
	double *angw = new double[Nang];
	
	// Working variables
	double *A = new double[N];
	double *B = new double[N];
	double *Anew = new double[N];
	double *Bnew = new double[N];
	double *angulardataA = new double[N*N];
	double *angulardataB = new double[N*N];
	double *epsA = new double[N];
	double *epsB = new double[N];

	// Initialize initial arrays
	for(int i=0;i<N;i++){
		A[i]=A0;
		B[i]=B0;
	}

	memcpy(Anew,A,sizeof(double)*N);
	memcpy(Bnew,B,sizeof(double)*N);

	// Calculate weights, nodes and remap
	cout << "\tCalculating weights, notes and remap on CPU... " << flush;
	gauleg(0,1,x,w,N);
	mapping(xmap, wmap, x, w, a, b, s, N);
	for(int j=0;j<Nang;j++){
		angx[j]=cos((j+1.0)/(Nang+2.0) * M_PI);
		angw[j]=M_PI/(Nang+2.0)*pow(sin((j+1.0)/(Nang+2.0)*M_PI),2);
	}
	cout << "done!" << endl;

	q.enqueueWriteBuffer(b_xmap, CL_TRUE, 0, N*sizeof(cl_double), xmap);
	q.enqueueWriteBuffer(b_wmap, CL_TRUE, 0, N*sizeof(cl_double), wmap);

	q.enqueueWriteBuffer(b_angx, CL_TRUE, 0, Nang*sizeof(cl_double), angx);
	q.enqueueWriteBuffer(b_angw, CL_TRUE, 0, Nang*sizeof(cl_double), angw);	

	double args[4];

	cout << "\tSetting kernel args... " << flush;
	ker.setArg(0, b_Anew);
	ker.setArg(1, b_Bnew);
	ker.setArg(2, b_A);
	ker.setArg(3, b_B);
	ker.setArg(4, b_angulardataA);
	ker.setArg(5, b_angulardataB);
	ker.setArg(6, b_xmap);
	ker.setArg(7, b_wmap);
	ker.setArg(8, m0);
	ker.setArg(9, D/(omega*omega));
	ker.setArg(10, N);
	ker.setArg(11, b_epsA);
	ker.setArg(12, b_epsB);
	angularker.setArg(0, b_angulardataA);
	angularker.setArg(1, b_angulardataB);
	angularker.setArg(2, b_xmap);
	angularker.setArg(3, b_angx);
	angularker.setArg(4, b_angw);
	// Argument 5 has to be set later as xi element
	angularker.setArg(6, omega);
	angularker.setArg(7, N);
	angularker.setArg(8, Nang);
	cout << "\t\tdone!" << endl;

	cout << "\tWriting buffers to GPU... " << flush;
	q.enqueueWriteBuffer(b_A, CL_TRUE, 0, N*sizeof(cl_double), A);
	q.enqueueWriteBuffer(b_B, CL_TRUE, 0, N*sizeof(cl_double), B);
	cout << "\tdone!" << endl;

	cout << "\tAngular integration on GPU... " << flush;
	angularker.setArg(5, 0);
	cl::Event eventang;
	q.enqueueNDRangeKernel(angularker, cl::NullRange, cl::NDRange(N), cl::NDRange(wgsize), NULL, &eventang);
	cout << "\tdone!" << endl << endl;
	
	double epsmaxA=0;
	double epsmaxB=0;

	for(int i=0;i<iter;i++){
		epsmaxA=0;
		epsmaxB=0;

		cout << "\t--- Iteration " << i+1 << " ---" << endl;

		cout << "\tRadial integration on GPU... " << flush;
		cl::Event event;
		q.enqueueNDRangeKernel(ker, cl::NullRange, cl::NDRange(N), cl::NDRange(wgsize), NULL, &event);
		cout << "\tdone!" << endl;

		cout << "\tCopying buffers around... " << flush;
		q.enqueueCopyBuffer(b_Anew,b_A,0,0,N*sizeof(cl_double));
		q.enqueueCopyBuffer(b_Bnew,b_B,0,0,N*sizeof(cl_double));
		cout << "\tdone!" << endl;
		
		cout << "\tCovergence check... " << endl;
		q.enqueueReadBuffer(b_epsA, CL_TRUE, 0, N*sizeof(cl_double), epsA);
		q.enqueueReadBuffer(b_epsB, CL_TRUE, 0, N*sizeof(cl_double), epsB);
	
		for(int i=0;i<N;i++){
			if(epsA[i]>epsmaxA)
				epsmaxA=epsA[i];	
			if(epsB[i]>epsmaxB)
				epsmaxB=epsB[i];	
		}
		
		cout << "\tChange in A: "<< epsmaxA << " Change in B: " << epsmaxB << endl << endl;

		if(epsmaxA<eps && epsmaxB<eps)
			break;


		q.finish();
	}

	cout << endl << "\tReading buffers from GPU... " << flush;
	// read output data
	q.enqueueReadBuffer(b_Anew, CL_TRUE, 0, N*sizeof(cl_double), A);
	q.enqueueReadBuffer(b_Bnew, CL_TRUE, 0, N*sizeof(cl_double), B);
	
	q.finish();
	
	cout << "\tdone!" << endl;

	ofstream fout;
	fout.open("dressing.data");

	for(int xi=0;xi<N;xi++){
		fout << xmap[xi] << " " << A[xi] << " " << B[xi] << " " << B[xi]/A[xi] << endl;
	}
	fout.close();

	cout << endl << endl << "Everything done!" << endl;

	return 0;
}

