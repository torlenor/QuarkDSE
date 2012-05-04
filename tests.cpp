#include <iostream>
#include <fstream>
#include <cmath>

#include "integration.hpp"

using namespace std;

double ftest(double *x){
	return sin(x[0])*sin(x[0]);
	// return x*x+ x + 3.0;
}

void reweight(double *xi, double *wi, double *xinew, double *winew){
	
}

int main(){
	ofstream fout;
	
	fout.open("int_convergence.data");

	// Function pointer
	double (*fx)(double*);
	fx=&ftest;

	int N=128, Nmax=1024;
	double *fData, simpsonsInt, gcheby;
	double xmin=-1.0;
	double h=0;

	double mmint=0.33244;

	double x[1];

	fout << 0 << " . . " <<  mmint << endl; 
	for(int N=2;N<Nmax+1;N=N*2){
		fData = new double[N+1];
		h=2.0/(double)N;

		for(int i=0;i<N+1;i++){
			x[0]=xmin + i*h;
			fData[i]=sqrt(1-pow(x[0],2))*fx(x);
		}

		simpsonsInt = IntSimpson(fData, h, N+1);
		gcheby = gausscheby(fx,x,0,N);

		cout << "N=" << N << " Gauss-Cheby: " << gcheby << " Simpsons: " << simpsonsInt << endl;
		fout << N << " " << gcheby << " " << simpsonsInt << " " << mmint << endl; 
	}
	fout.close();

	cout << endl << "Mathematica result: " << 0.33244 << endl;
}
