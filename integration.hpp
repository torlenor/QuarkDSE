#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <iostream>
#include <cmath>

using namespace std;

template <class T>
T IntSimpson(T *data, double h, int nData){
        // extended Simpsons rule
        T intvalue=0;
        intvalue=data[0]/3.0;
        for(int k=1;k<nData-1;k++){
                if(k % 2 == 0){ 
                        intvalue=intvalue+2.0*data[k]/3.0;
                }else{
                        intvalue=intvalue+4.0*data[k]/3.0;
                }   
        }   
        intvalue=intvalue + data[nData-1]/3.0;
        return intvalue*h;
}

double gausscheby(double (*fx)(double), int N){
	// Integration of sqrt(1-x^2)*fx(x)
	
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
