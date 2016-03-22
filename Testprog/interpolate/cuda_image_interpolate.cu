/*
 * $Id: cuda_image_interpolate.cu,v 1.2 2012-08-30 12:19:21 ueshiba Exp $
 */
#include "TU/cuda/Array++.h"
#include "TU/Image++.h"
#include <cutil.h>
#include <cutil_math.h>

namespace TU
{
namespace cuda
{
template <class T> __device__ T
interpolate_pixel(T s0, T s1, float r0, float r1)
{
    return s0 * r0 + s1 * r1;
}

#if 0
template <> __device__ RGBA
interpolate_pixel(RGBA s0, RGBA s1, float r0, float r1)
{
    RGBA	val;
    val.r = s0.r * r0 + s1.r * r1;
    val.g = s0.g * r0 + s1.g * r1;
    val.b = s0.b * r0 + s1.b * r1;
    
    return val;
}
#endif
    
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
    
  // timer
    u_int	timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));

  // setup execution parameters
    dim3  threads(16, 16, 1);
    dim3  blocks(d_image0.ncol()/threads.x, d_image0.nrow()/threads.y, 1);
    cerr << blocks.x << 'x' << blocks.y << " blocks..." << endl;
    
  // execute the kernel
    cerr << "Let's go!" << endl;
    for (int i = 0; i < 1000; ++i)
	interpolate_kernel<<<blocks, threads>>>(d_image0.data().get(),
						d_image1.data().get(),
						d_image2.data().get(),
						d_image2.stride(), 0.5f);
    cerr << "Returned!" << endl;
    
  // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

  // time
    CUT_SAFE_CALL(cutStopTimer(timer));
    cerr << "Processing time: " << cutGetTimerValue(timer) << " (ms)" << endl;
    CUT_SAFE_CALL(cutDeleteTimer(timer));
}

template void	interpolate(const Array2<u_char>& d_image0,
			    const Array2<u_char>& d_image1,
				  Array2<u_char>& d_image2)	;
  /*
template void	interpolate(const Array2<RGBA>&   d_image0,
			    const Array2<RGBA>&   d_image1,
				  Array2<RGBA>&   d_image2)	;
  */
template void	interpolate(const Array2<float4>& d_image0,
			    const Array2<float4>& d_image1,
				  Array2<float4>& d_image2)	;
}
}
