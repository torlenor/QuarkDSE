#include <iostream>
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
	// Function pointer
	double (*fx)(double);
	fx=&ftest;

	int N=60;

	cout << gausscheby(fx,N) << endl;

}
