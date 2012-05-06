#ifndef CPUFUNCTS_HPP
#define CPUFUNCTS_HPP

#include <iostream>
#include <cmath>

using namespace std;

void gauleg(const float x1, const float x2, float *x, float *w, int N){
// Given the lower and upper limits of integration x1 and x2, this routine returns arrays x[0..n-1] and w[0..n-1] of length n, containing the abscissas and weights of the Gauss-Legendre n-point quadrature formula.
const double EPS=1.0e-14; // EPS is the relative precision.
double z1,z,xm,xl,pp,p3,p2,p1;
int n=N;
int m=(n+1)/2;
int iterMax=1000, iter=0;

//The roots are symmetric in the interval, so
xm=0.5*(x2+x1);

// we only have to find half of them.
xl=0.5*(x2-x1);

for (int i=0;i<m;i++) { //Loop over the desired roots.
	// Starting with this approximation to the ith root, we enter the main loop of refinement by Newton’s method.
	z=cos(M_PI*(i+0.75)/(n+0.5)); 
	do {
		p1=1.0;
		p2=0.0;
		for (int j=0;j<n;j++) { // Loop up the recurrence relation to get the Legendre polynomial evaluated at z.
			p3=p2;
			p2=p1;
			p1=((2.0*j+1.0)*z*p2-j*p3)/(j+1);
		}
		// p1 is now the desired Legendre polynomial. We next compute pp, its derivative,
		// by a standard relation involving also p2, the polynomial of one lower order.
		pp=n*(z*p1-p2)/(z*z-1.0);
		z1=z;
		z=z1-p1/pp; //Newton’s method.

		iter++;
	} while (abs(z-z1) > EPS && iter < iterMax);
	x[i]=xm-xl*z; // Scale the root to the desired interval,
	x[n-1-i]=xm+xl*z; // and put in its symmetric counterpart.
	w[i]=2.0*xl/((1.0-z*z)*pp*pp); // Compute the weight
	w[n-1-i]=w[i]; // and its symmetric counterpart.
	}
}

template <class T>
T IntSimpson(T *data, float h, int nData){
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

/* float gausscheby(float (*fx)(float), int N){
	// integration of sqrt(1-x^2)*fx(x)
	
	float x[N], w[N];
	// build weights and x array
	for(int i=0;i<N;i++){
		x[i]=cos((float)(i+1)/(float)(N+1+1) * M_PI);
		w[i]=M_PI/(float)(N+1+1)*pow(sin((float)(i+1)/(float)(N+1+1)*M_PI),2);
	}

	float sum=0;
	for(int n=0;n<N;n++){
		sum += w[n]*fx(x[n]);
	}

	return sum;
} */

float gausscheby(float (*fx)(float*), float *args, int i, int N){
	// integration of sqrt(1-x^2)*fx(x)
	
	float x[N], w[N];
	// build weights and x array
	for(int j=0;j<N;j++){
		x[j]=cos((float)(j+1)/(float)(N+1+1) * M_PI);
		w[j]=M_PI/(float)(N+1+1)*pow(sin((float)(j+1)/(float)(N+1+1)*M_PI),2);
	}

	float sum=0;
	for(int n=0;n<N;n++){
		args[i]=x[n];
		sum += w[n]*fx(args);
	}

	return sum;
}

#endif /* CPUFUNCTS_HPP */
