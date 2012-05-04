#include <iostream>
#include <fstream>
#include <cmath>

#include "integration.hpp"

using namespace std;

double ftest(double x){
	return sin(x)*sin(x);
	// return x*x+ x + 3.0;
}

void reweight(double *xi, double *wi, double *xinew, double *winew){
	
}

int main(){
	ofstream fout;
	
	fout.open("int_convergence.data");

	// Function pointer
	double (*fx)(double);
	fx=&ftest;

	int N=128, Nmax=1024;
	double *fData, simpsonsInt, gcheby;
	double xmin=-1.0;
	double h=0;

	double mmint=0.33244;

	fout << 0 << " . . " <<  mmint << endl; 
	for(int N=2;N<Nmax+1;N=N*2){
		fData = new double[N+1];
		h=2.0/(double)N;

		for(int i=0;i<N+1;i++){
			fData[i]=sqrt(1-pow(xmin + i*h,2))*fx(xmin + i*h);
		}

		simpsonsInt = IntSimpson(fData, h, N+1);
		gcheby = gausscheby(fx,N);

		cout << "N=" << N << " Gauss-Cheby: " << gcheby << " Simpsons: " << simpsonsInt << endl;
		fout << N << " " << gcheby << " " << simpsonsInt << " " << mmint << endl; 
	}
	fout.close();

	cout << endl << "Mathematica result: " << 0.33244 << endl;
}
