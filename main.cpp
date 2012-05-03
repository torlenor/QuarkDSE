#include <iostream>
#include <cmath>

#include "integration.hpp"

using namespace std;

double ftest(double x){
	return x*x+ x + 3.0;
}

int main(){
	// Function pointer
	double (*fx)(double);
	fx=&ftest;

	int N=100;

	cout << fx(1) << endl;
	cout << gausscheby(fx,N) << endl;

}
