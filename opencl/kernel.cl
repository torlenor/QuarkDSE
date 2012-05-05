/*!
 * Sample kernel which multiplies every element of the input array with
 * a constant and stores it at the corresponding output array
 */

__kernel void templateKernel(__global float * output,
                             __global float * inputA,
                             __global float * inputB)
{
    uint tid = get_global_id(0);
    
    output[tid] = inputA[tid] * inputB[tid];
}

__kernel void gausschebyKernel(__global float *output, uint N){
        // integration of sqrt(1-x^2)*fx(x)
    	uint tid = get_global_id(0);
	float x, w;
        
	x=native_cos((float)((tid+1.0)/(N+1.0+1.0) * M_PI));
	w=M_PI/(float)(N+1+1)*pow(native_sin((float)((tid+1.0)/(N+1.0+1.0)*M_PI)),2);

	output[tid] = w*pow(native_sin(x),2);
}
