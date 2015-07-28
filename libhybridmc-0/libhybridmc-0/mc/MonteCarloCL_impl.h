#ifndef MONTECARLO_CL_IMPL_H
#define MONTECARLO_CL_IMPL_H

#include <CL/opencl.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <unistd.h>  //gwtcwd

static cl_context cxGPUContext;        // OpenCL context
static cl_command_queue cqCommandQueue;// OpenCL command que
static cl_platform_id cpPlatform;      // OpenCL platform
static cl_device_id cdDevice;          // OpenCL device
static cl_program cpProgram;           // OpenCL program
static cl_kernel ckKernel;             // OpenCL kernel
static cl_mem dev_D;
static cl_mem dev_x;
static cl_mem dev_estimation;
static cl_int ciErr1;			// Error code var

const char* descriptionOfError(int err) {
    switch (err) {
    case CL_SUCCESS:                            return "Success!";
    case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                   return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
    case CL_MAP_FAILURE:                        return "Map failure";
    case CL_INVALID_VALUE:                      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
    case CL_INVALID_PLATFORM:                   return "Invalid platform";
    case CL_INVALID_DEVICE:                     return "Invalid device";
    case CL_INVALID_CONTEXT:                    return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
    case CL_INVALID_SAMPLER:                    return "Invalid sampler";
    case CL_INVALID_BINARY:                     return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
    case CL_INVALID_PROGRAM:                    return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                     return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
    case CL_INVALID_EVENT:                      return "Invalid event";
    case CL_INVALID_OPERATION:                  return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
    default: return "Unknown";
    }
}

static std::string create_function_definition(std::string name, std::string cpp,
                                              bool _inline = true) {
    std::string f;
    if (_inline)
        f += "inline ";
    f += "double " + name + "(double *x) { return " + cpp + "; }\n";
    return f;
}

cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID,
                        const char *platform_name)
{
    char chBuffer[1024];
    cl_uint num_platforms;
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;

    // Get OpenCL platform count
    ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS)
        {
            return -1000;
        }
    else
        {
            if(num_platforms == 0)
                {
                    return -2000;
                }
            else
                {
                    // if there's a platform or more, make space for ID's
                    if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
                        {
                            return -3000;
                        }

                    // get platform info for each platform and trap the NVIDIA platform if found
                    ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
                    if ( platform_name )
                        for(cl_uint i = 0; i < num_platforms; ++i)
                            {
                                ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                                if(ciErrNum == CL_SUCCESS)
                                    {
                                        if(strcasestr(chBuffer, platform_name) != NULL)
                                            {
                                                *clSelectedPlatformID = clPlatformIDs[i];
                                                break;
                                            }
                                    }
                            }

                    // default to zeroeth platform if NVIDIA not found
                    if(*clSelectedPlatformID == NULL)
                        {
                            *clSelectedPlatformID = clPlatformIDs[0];
                        }

                    free(clPlatformIDs);
                }
        }

    return CL_SUCCESS;
}

template<int dim>
std::string MonteCarloCL<dim>::load_program(const std::string file)
{
    std::cout << "Kernel: '";
#if 0
    if (file[0] != '/') {
        char cCurrentPath[FILENAME_MAX];
        if (!getcwd(cCurrentPath, sizeof(cCurrentPath))) {
            std::perror("pwd");
            return std::string();
        }
        cCurrentPath[sizeof(cCurrentPath) - 1] = '\0'; /* not really required */
        std::cout << cCurrentPath << "/";
    }
#endif

    std::cout << file << "'" << std::endl;

    std::ifstream f;
    std::string line;

    std::string OpenCL_ExtensionPragmas = "\n#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";

    std::string f_expr = create_function_definition("CPPCODE_F",pp.cppcode_f);
    std::string q_expr = create_function_definition("CPPCODE_Q",pp.cppcode_q);
    std::string out = OpenCL_ExtensionPragmas + f_expr + q_expr;

    f.open(file);

    if ( f.is_open() ) {
        while( f.good() ) {
            std::getline(f, line);
            out += line + "\n";
        }
        f.close();
    }
    else
        std::perror("error");

    return out;
}

template<>
bool MonteCarloCL<3>::init()
{
    size_t size;
    size_t ret;
    char cBuffer[1024];
    ciErr1  = oclGetPlatformID(&cpPlatform,NULL);
    ciErr1 |=clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ALL, 1, &cdDevice, NULL);
    if ( ciErr1 != CL_SUCCESS ) {
        assert(0);
    }
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    if ( ciErr1 != CL_SUCCESS ) {
        assert(0);
    }
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    if ( ciErr1 != CL_SUCCESS ) {
        assert(0);
    }
    std::string cSourceCL = load_program(pp.clkernel);
    size_t szKernelLength = cSourceCL.length();

    clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
    std::cout << "DEVICE INFO: " << cBuffer << std::endl;
    const char *ptr = cSourceCL.c_str();
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
                                          &ptr, &szKernelLength, &ciErr1);

    ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if ( ciErr1 != CL_SUCCESS ) {
        size_t ret_val_size;
        char *build_log;
        clGetProgramBuildInfo(cpProgram,  cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        build_log = (char*) malloc(sizeof(char)*ret_val_size);

        clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = 0;
        std::cout << "ERROR: " << build_log << std::endl;
        exit(0);

    }

    ckKernel = clCreateKernel(cpProgram, "DoRandomWalks3D", &ciErr1);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}

    size = sizeof(size_t);
    ciErr1 = clGetKernelWorkGroupInfo(ckKernel, cdDevice,  CL_KERNEL_WORK_GROUP_SIZE , size, &ret, NULL );
    if ( ciErr1!= CL_SUCCESS ) {
        assert(0);
    }
    if ( ret > 768) {
        ret = 768;
    }
    set_workgroup(ret);
    return true;
}

template <>
std::vector<double> MonteCarloCL<3>::execute()
{
    //double *p_node_coord = (double*)(&node_coord[nof_nodes-1])+1;
    double *p_node_coord = node_coord;
    int nof_walks;
    static size_t szGlobalWorkSize;        // 1D var for Total # of work items
    static size_t szLocalWorkSize;		    // 1D var for # of work items in the work group

    szLocalWorkSize  = max_workgroup;
    szGlobalWorkSize = nof_nodes*max_workgroup;

    nof_walks = pp.nof_walks/max_workgroup;
    if ( pp.nof_walks % max_workgroup ) {
        nof_walks ++;
    }

    std::cout << "NUMBER OF WALKS: " << nof_walks << std::endl;
    std::cout << "NOF NODES: " << nof_nodes << std::endl;

    dev_x = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_double)*nof_nodes*3, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    dev_D = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_double)*3, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    dev_estimation = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_double)*nof_nodes, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}

    ciErr1  = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&dev_D);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}

    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&dev_x);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&dev_estimation);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&nof_walks);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_double), (void*)&pp.btol);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_int), (void*)&nof_nodes);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}

    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, dev_D, CL_TRUE, 0, sizeof(cl_double) * 3, pp.D, 0, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, dev_x, CL_TRUE, 0, sizeof(cl_double) *nof_nodes* 3, p_node_coord, 0, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    // Launch kernel
    ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1,
                                    NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
	{
            std::cout << "global: " << szGlobalWorkSize << " local: " << szLocalWorkSize << std::endl;
            std::cout << "ERROR: " << ciErr1 << "::: " << descriptionOfError(ciErr1) << std::endl;
            assert(0);
	}

    std::vector<double> estimation(nof_nodes);
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, dev_estimation, CL_TRUE, 0, sizeof(cl_double) * nof_nodes, estimation.data(), 0, NULL, NULL);

    if (ciErr1 != CL_SUCCESS)
	{
            std::cout << descriptionOfError(ciErr1) << std::endl;
            assert(0);
	}
    return estimation;
}

template<>
bool MonteCarloCL<2>::init()
{
    size_t size;
    size_t ret;
    cl_uint num;
    char cBuffer[1024];
    std::cout << "GetPlatformID" << ciErr1 << std::endl << std::flush;
    ciErr1  = oclGetPlatformID(&cpPlatform,NULL);
    ciErr1 |=clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ALL, 1, &cdDevice, &num);
    if ( ciErr1 != CL_SUCCESS ) {
        assert(0);
    }
    std::cout << "GetDeviceIDS" << ciErr1 << std::endl << std::flush;
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    if ( ciErr1 != CL_SUCCESS ) {
        assert(0);
    }
    std::cout << "CreateContext" << ciErr1 << std::endl << std::flush;
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    if ( ciErr1 != CL_SUCCESS ) {
        assert(0);
    }
    std::cout << "CreateCommandQueue " << ciErr1 << std::endl << std::flush;
    std::string cSourceCL = load_program(pp.clkernel);
    size_t szKernelLength = cSourceCL.length();
    std::cout << "LoadProgram " << szKernelLength << std::endl << std::flush;
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
    std::cout << "ClGetDeviceInfo " << ciErr1 << std::endl << std::flush;
    std::cout << "DEVICE INFO: " << cBuffer << std::endl;

    const char *ptr = cSourceCL.c_str();
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
                                          &ptr, &szKernelLength, &ciErr1);
    if ( ciErr1 != CL_SUCCESS ) {
        std::cout << descriptionOfError(ciErr1) << std::endl;
        exit(0);
    }

    ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if ( ciErr1 != CL_SUCCESS ) {
        size_t ret_val_size;
        char *build_log;
        clGetProgramBuildInfo(cpProgram,  cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        build_log = (char*) malloc(sizeof(char)*ret_val_size);

        clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = 0;
        std::cout << "ERROR: " << build_log << std::endl;
        free(build_log);
        exit(0);

    }

    ckKernel = clCreateKernel(cpProgram, "DoRandomWalks2D", &ciErr1);
    if (ciErr1 != CL_SUCCESS)
	{
            std::cout << descriptionOfError(ciErr1) << std::endl;
            assert(0);
	}

    size = sizeof(size_t);
    ciErr1 = clGetKernelWorkGroupInfo(ckKernel, cdDevice,  CL_KERNEL_WORK_GROUP_SIZE , size, &ret, NULL );
    if ( ciErr1!= CL_SUCCESS ) {
        assert(0);
    }
    if ( ret > 1024 ) {
        ret = 1024;
    }
    set_workgroup(ret);
    return true;
}

template <>
std::vector<double> MonteCarloCL<2>::execute()
{
    //double *p_node_coord = (double*)(&node_coord[nof_nodes-1])+1;
    double *p_node_coord = node_coord;
    int nof_walks;
    static size_t szGlobalWorkSize;        // 1D var for Total # of work items
    static size_t szLocalWorkSize;		    // 1D var for # of work items in the work group

    szLocalWorkSize  = max_workgroup;
    szGlobalWorkSize = nof_nodes*max_workgroup;

    nof_walks = pp.nof_walks/max_workgroup;
    if ( pp.nof_walks % max_workgroup ) {
        nof_walks ++;
    }

    std::cout << "NUMBER OF WALKS: " << nof_walks << std::endl << std::flush;
    std::cout << "NOF NODES: " << nof_nodes << std::endl << std::flush;

    dev_x = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_double)*nof_nodes*2, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    dev_D = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_double)*2, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    dev_estimation = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_double)*nof_nodes, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}

    ciErr1  = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&dev_D);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}

    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&dev_x);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&dev_estimation);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&nof_walks);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_double), (void*)&pp.btol);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_int), (void*)&nof_nodes);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}

    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, dev_D, CL_TRUE, 0, sizeof(cl_double) * 2, pp.D, 0, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, dev_x, CL_TRUE, 0, sizeof(cl_double) *nof_nodes* 2, p_node_coord, 0, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
	{
            assert(0);
	}
    // Launch kernel
    ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1,
                                    NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
	{
            std::cout << descriptionOfError(ciErr1) << std::endl;
            assert(0);
	}

    std::vector<double> estimation(nof_nodes);
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, dev_estimation, CL_TRUE, 0, sizeof(cl_double) * nof_nodes, estimation.data(), 0, NULL, NULL);

    if (ciErr1 != CL_SUCCESS)
	{
            std::cout << descriptionOfError(ciErr1) << std::endl;
            assert(0);
	}
    return estimation;
}

#endif
