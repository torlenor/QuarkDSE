#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <iostream>
#include <cmath>

using namespace std;

double gausscheby(double (*fx)(double), int N){
	double x[N], w[N];
	// build weights and x array
	for(int i=0;i<N;i++){
		x[i]=cos((double)(i+1)/(double)(N+1+1) * M_PI);
		w[i]=M_PI/(double)(N+1+1)*pow(sin((double)(i+1)/(double)(N+1+1)*M_PI),2);
	}

	double sum=0;
	for(int n=0;n<N;n++){
		sum += w[n]*fx(x[n]);
	}

	return sum;
}

#endif /* INTEGRATION_HPP */
