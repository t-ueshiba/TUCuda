/*
 * $Id: cuda_image_interpolate.cu,v 1.2 2012-08-30 12:19:21 ueshiba Exp $
 */
#include "TU/cuda/Array++.h"
#include "TU/Image++.h"
#include <cutil.h>
#include <cutil_math.h>

namespace TU
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
interpolate(const Image<T>& image0, const Image<T>& image1, Image<T>& image2)
{
    using namespace	std;

  // timer
    u_int	timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));

  // allocate device memory and copy host memory to them
    CudaArray2<T>	d_image0(image0), d_image1(image1),
			d_image2(image0.height(), image0.width());
    
  // setup execution parameters
    dim3  threads(16, 16, 1);
    dim3  blocks(image0.width()/threads.x, image0.height()/threads.y, 1);
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

  // copy result from device to host
    d_image2.write(image2);

  // time
    CUT_SAFE_CALL(cutStopTimer(timer));
    cerr << "Processing time: " << cutGetTimerValue(timer) << " (ms)" << endl;
    CUT_SAFE_CALL(cutDeleteTimer(timer));
}

template void	interpolate(const Image<u_char>& image0,
			    const Image<u_char>& image1,
				  Image<u_char>& image2)	;
  /*
template void	interpolate(const Image<RGBA>&   image0,
			    const Image<RGBA>&   image1,
				  Image<RGBA>&   image2)	;
  */
template void	interpolate(const Image<float4>& image0,
			    const Image<float4>& image1,
				  Image<float4>& image2)	;
}
