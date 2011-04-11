/*
 *  $Id: cuda_image_interpolate_kernel.h,v 1.2 2011-04-11 08:06:06 ueshiba Exp $
 */
#ifndef __CUDA_IMAGE_INTERPOLATE_KERNEL_H__
#define __CUDA_IMAGE_INTERPOLATE_KERNEL_H__

__global__ void
interpolate_kernel(const TU::RGBA* src0, const TU::RGBA* src1, TU::RGBA* dst,
		   uint stride, float ratio)
{
    using namespace	TU;
    
    const uint	x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint	y = blockIdx.y * blockDim.y + threadIdx.y;
    
    const uint	n = x + y * stride;

    const float	r1 = ratio;
    const float	r0 = 1.0f - r1;
    
    const RGBA	s0 = src0[n];
    const RGBA	s1 = src1[n];

    dst[n].r = s0.r * r0 + s1.r * r1;
    dst[n].g = s0.g * r0 + s1.g * r1;
    dst[n].b = s0.b * r0 + s1.b * r1;
}

#endif // #ifndef __CUDA_IMAGE_INTERPOLATE_KERNEL_CU__
