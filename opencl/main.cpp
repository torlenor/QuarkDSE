#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "kernel.h"

using namespace std;

std::vector<cl::Platform> pl;
std::vector<cl::Device> devs;
cl::Context ctx;
cl::CommandQueue q;
cl::Program prog;
cl::Kernel ker;
cl::Buffer b_inputA, b_inputB, b_output;
int current_device;

void SetupDevice(unsigned int ip, unsigned int id);
void LoadOpenCLKernel(const char *kernelfile);

int main(int narg, char **argv)
{
  // select platform and device
  if (narg != 3) {
    SetupDevice(0, 0);
  } else {
    SetupDevice(atoi(argv[1]), atoi(argv[2]));
  }

  // load kernel string
  LoadOpenCLKernel(kernel_str);

  // Runtime paramters
  int nSize = pow(2,20); // size of arrays for scalar product
  std::cout << nSize << std::endl;

  // select a main kernel function
  ker = cl::Kernel(prog, "templateKernel");

  // allocate two OpenCL buffers
  b_output = cl::Buffer(ctx, CL_MEM_READ_WRITE, nSize*sizeof(cl_float));
  b_inputA = cl::Buffer(ctx, CL_MEM_READ_WRITE, nSize*sizeof(cl_float));
  b_inputB = cl::Buffer(ctx, CL_MEM_READ_WRITE, nSize*sizeof(cl_float));

  // map arguments
  ker.setArg(0, b_output);
  ker.setArg(1, b_inputA);
  ker.setArg(2, b_inputB);

  // set "multiplier"
//  unsigned int cc = 100;
//  ker.setArg(2, (unsigned int)cc);

  // write input data
  float *inA = new float[nSize];
  float *inB = new float[nSize];
  for(int i = 0; i < nSize; i++) {
    inA[i] = exp(-i/(double)100);
    inB[i] = i + 1.0/(i+1);
  }
  q.enqueueWriteBuffer(b_inputA, CL_TRUE, 0, nSize*sizeof(cl_float), inA);
  q.enqueueWriteBuffer(b_inputB, CL_TRUE, 0, nSize*sizeof(cl_float), inB);

  cout << CL_DEVICE_MAX_WORK_GROUP_SIZE << endl;

  // execute the kernel
  double iter=1000;
  cl::Event event;
  for(int i=0;i<iter;i++){
	  q.enqueueNDRangeKernel(ker, cl::NullRange, cl::NDRange(nSize), cl::NDRange(128), NULL, &event);
  }

  // read output data
  float *out = new float[nSize];
  q.enqueueReadBuffer(b_output, CL_TRUE, 0, nSize*sizeof(cl_float), out);

  // verify results
  float sum=0, sumhost=0;
  for(int i = 0; i < nSize; i++) {
	  sum+=out[i];
	  sumhost+=inA[i]*inB[i];
  }
  std::cout << sum << " " << sumhost << std::endl; 
}

void SetupDevice(unsigned int ip, unsigned int id)
{
  try {
    cl::Platform::get(&pl);

    for(unsigned int i = 0; i < pl.size(); i++) {
      std::cerr << "platform " << i << " " << pl[i].getInfo<CL_PLATFORM_NAME>().c_str() << " "
	      << pl[i].getInfo<CL_PLATFORM_VERSION>().c_str() << "\n";    
      pl[i].getDevices(CL_DEVICE_TYPE_ALL, &devs);
      for(unsigned int j = 0; j < devs.size(); j++) {
	std::cerr << "\tdevice " << j << " " << devs[j].getInfo<CL_DEVICE_NAME>().c_str() << "\n";
      }

      cl::Context context(CL_DEVICE_TYPE_GPU);
      std::vector<cl::Device> devs = context.getInfo<CL_CONTEXT_DEVICES>();
      std::cerr << "I Found " << devs.size() << "\n";

      cl::Context ctx2 = cl::Context(devs);
      std::vector<cl::Device> devs2 = ctx2.getInfo<CL_CONTEXT_DEVICES>();
      std::cerr << "I Found " << devs2.size() << "\n";
    }
    std::cerr << "\n"; 

    if (pl.size() <= ip) throw cl::Error(-1, "FATAL: the specifed platform does not exist");
    std::cerr << pl[ip].getInfo<CL_PLATFORM_NAME>().c_str() << " "
	      << pl[ip].getInfo<CL_PLATFORM_VERSION>().c_str() << "::";    

    pl[ip].getDevices(CL_DEVICE_TYPE_ALL, &devs);
    if (devs.size() <= id) throw cl::Error(-1, "FATAL: the specifed device does not exist");
    std::cerr << devs[id].getInfo<CL_DEVICE_NAME>().c_str() << "\n";

    ctx = cl::Context(devs);
    q = cl::CommandQueue(ctx, devs[id], CL_QUEUE_PROFILING_ENABLE);
  }  catch( cl::Error e ) {
    std::cerr << e.what() << ":" << e.err() << "\n";
    std::cerr << "Abort!\n";
    exit(-1);
  }
  current_device = id;
}

void LoadOpenCLKernel(const char *kernelfile) 
{
  try {
    cl::Program::Sources src(1, std::make_pair(kernelfile, strlen(kernelfile)));
    prog = cl::Program(ctx, src);

    std::stringstream options;
    options << " -D__DUMMY_BUILD_OPTIONS__  ";
    std::cerr << "Build options :: " << options.str() << "\n";
    prog.build(devs, options.str().c_str());
  }  catch( cl::Error e ) {
    std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devs[current_device]);
    std::cerr << e.what() << ":" << e.err() << "\n";
    std::cerr << kernelfile << "\n";
    std::cerr << log << "\n";
    exit(-1);
  }
}
