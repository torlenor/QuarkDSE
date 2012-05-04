#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

#include <gsl/gsl_sf_legendre.h>

#include "integration.hpp"

using namespace std;

double ftest(double x){
	return sin(x)*sin(x);
}

void mapping(double *xmap, double *wmap, double *x, double *w, double a, double b, double s, int N){
	double g=log(1.0 + (b-a)/s);
	for(int i=0;i<N;i++){
		xmap[i]=a + s*(exp(g*x[i]) - 1.0)/(1.0 + exp(1) - exp(x[i]));
		wmap[i]=w[i]*(s*g*exp(g*x[i]) + (xmap[i] - a)*exp(x[i]))/(1.0+exp(1.0)-exp(x[i]));
	}
}

double angularA(double *args){
	double x=args[0];
	double y=args[1];
	double z=args[2];
	double w=args[3];

	return 2.0/M_PI * (-2.0/3.0*y + (1 + y/x)*sqrt(x*y)*z - 4.0/3.0*y*z*z)
		*exp(-(x+y-2*sqrt(x*y)*z)/(w*w));
}
double angularB(double *args){
	double x=args[0];
	double y=args[1];
	double z=args[2];
	double w=args[3];

	return 2.0/M_PI * (x + y - 2.0*sqrt(x*y)*z)
		*exp(-(x+y-2.0*sqrt(x*y)*z)/(w*w));
}

int main(int argc, char *argv[]){
	// Parameters
	double a=1E-4, b=5E4; // IR/UV cutoff
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

	int iter=16; // How many iterations
	int N=256; // Number of discretized values for integration
	double s=1; // Mapping parameter

	// Write parameters to stdout
	cout << "Quark DSE Solver on CPU v1.0" << endl;
	cout << "(c) Hans-Peter Schadler" << endl << endl;
	cout << "Physical parameter" << endl;
	cout << "IR/UV cutoff: a=" << a << " b=" << b << endl;
	cout << "D=" << D << endl;
	cout << "Omega=" << omega << endl;
	cout << "m0=" << m0 << endl << endl;
	cout << "Numerical parameter" << endl;
	cout << "Initial values: A0=" << A0 << " B0=" << B0 << endl;
	cout << "Number of iterations: " << iter << endl;
	cout << "Number of integration points: " << N << endl << endl;

	cout << "Starting calculation... " << flush;

	// Integraten nodes and weights
	double *xmap, *w, *wmap, *x, *dtmpa;
	x = new double[N];
	xmap = new double[N];
	w = new double[N];
	wmap = new double[N];
	dtmpa = new double[N];
	
	// Working variables
	double *A, *Anew;
	double *B, *Bnew;
	A = new double[N];
	B = new double[N];
	Anew = new double[N];
	Bnew = new double[N];
	double angulardataA[N][N];
	double angulardataB[N][N];

	// Initialize initial arrays
	for(int i=0;i<N;i++){
		A[i]=A0;
		B[i]=B0;
	}

	memcpy(Anew,A,sizeof(double)*N);
	memcpy(Bnew,B,sizeof(double)*N);

	// Calculate weights, nodes and remap
	// gauleg(0,1,x,w,N);
	// gauleg(a,b,xmap,wmap,N);
	gauleg(0,1,x,w,N);
	// gauleg(-1,1,dtmpa,w,N);
	mapping(xmap, wmap, x, w, a, b, s, N);
	
	double args[4];

	for(int i=0;i<iter;i++){
		for(int xi=0;xi<N;xi++)
			for(int yi=0;yi<N;yi++){
				args[0]=xmap[xi];
				args[1]=xmap[yi];
				args[2]=0;
				args[3]=omega;
				angulardataA[xi][yi]=gausscheby(angularA, args, 2, N);
				angulardataB[xi][yi]=gausscheby(angularB, args, 2, N);
			}

		// Gauss-Legendre Integration for A(x), B(x)
		for(int xi=0;xi<N;xi++){
			Anew[xi]=1.0;
			Bnew[xi]=m0;
			for(int yi=0;yi<N;yi++){
				Anew[xi]+=wmap[yi]*D/(omega*omega)*(xmap[yi]*A[yi]/(xmap[yi]*A[yi]*A[yi]+B[yi]*B[yi])
						*angulardataA[xi][yi]);
				Bnew[xi]+=wmap[yi]*D/(omega*omega)*(xmap[yi]*B[yi]/(xmap[yi]*A[yi]*A[yi]+B[yi]*B[yi])
						*angulardataB[xi][yi]);
			}
		}

		memcpy(A,Anew,sizeof(double)*N);
		memcpy(B,Bnew,sizeof(double)*N);
	}

	ofstream fout;
	fout.open("dressing.data");

	for(int xi=0;xi<N;xi++){
		fout << xmap[xi] << " " << A[xi] << " " << B[xi] << " " << B[xi]/A[xi] << endl;
	}
	fout.close();

	cout << "done" << endl;

	return 0;
}

