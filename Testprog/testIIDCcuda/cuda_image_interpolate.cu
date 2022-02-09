/*
 * $Id: cuda_image_interpolate.cu,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cu/Array++.h"
#include "TU/cu/chrono.h"

namespace TU
{
namespace cu
{
template <class T> __device__ T
interpolate_pixel(T s0, T s1, float r0, float r1)
{
    return s0 * r0 + s1 * r1;
}
  /*    
template <> __device__ RGBA
interpolate_pixel(RGBA s0, RGBA s1, float r0, float r1)
{
    RGBA	val;
    val.r = s0.r * r0 + s1.r * r1;
    val.g = s0.g * r0 + s1.g * r1;
    val.b = s0.b * r0 + s1.b * r1;
    
    return val;
}
  */
template <class T> __global__ void
interpolate_kernel(const T* src0, const T* src1, T* dst,
		   u_int stride, float ratio)
{
    const u_int	xy = (blockIdx.y * blockDim.y + threadIdx.y) * stride
		   +  blockIdx.x * blockDim.x + threadIdx.x;

    dst[xy] = interpolate_pixel(src0[xy], src1[xy], ratio, 1.0f - ratio);
}
    
template <class T> void
interpolate(const Array2<T>& d_image0,
	    const Array2<T>& d_image1, Array2<T>& d_image2)
{
    using namespace	std;

    d_image2.resize(d_image0.nrow(), d_image0.ncol());
    
#ifdef PROFILE
    Profiler<clock>	cuProfiler(1);
    cuProfiler.start(0);
#endif
  // setup execution parameters
    dim3  threads(16, 16, 1);
    dim3  blocks(d_image0.ncol()/threads.x, d_image0.nrow()/threads.y, 1);
    
  // execute the kernel
    interpolate_kernel<<<blocks, threads>>>(d_image0.data().get(),
					    d_image1.data().get(),
					    d_image2.data().get(),
					    d_image2.stride(), 0.5f);
#ifdef PROFILE
    cuProfiler.stop();
    cuProfiler.nextFrame();
    cuProfiler.print(cerr);
#endif
}

template void	interpolate(const Array2<u_char>& d_image0,
			    const Array2<u_char>& d_image1,
				  Array2<u_char>& d_image2)	;
  /*
template void	interpolate(const Array2<RGBA>&   d_image0,
			    const Array2<RGBA>&   d_image1,
				  Array2<RGBA>&   d_image2)	;
  */
}
}
