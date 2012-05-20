#ifndef GPUFUNCTS_HPP
#define GPUFUNCTS_HPP

// Global OpenCL variables
std::vector<cl::Platform> pl;
std::vector<cl::Device> devs;
cl::Context ctx;
cl::CommandQueue q;
cl::Program prog;
cl::Kernel ker, angularker;
int current_device;

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

     //  cl::Context context(CL_DEVICE_TYPE_CPU);
     // std::vector<cl::Device> devs = context.getInfo<CL_CONTEXT_DEVICES>();
     // std::cerr << "I Found " << devs.size() << "\n";

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
    // q = cl::CommandQueue(ctx, devs[id], CL_QUEUE_PROFILING_ENABLE);
    q = cl::CommandQueue(ctx, devs[id]);
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

#endif /* GPUFUNCTS_HPP */
