/*
 *  $Id: cuda_image_interpolate_kernel.h,v 1.1 2009-04-15 00:32:26 ueshiba Exp $
 */
#ifndef __CUDA_IMAGE_INTERPOLATE_KERNEL_H__
#define __CUDA_IMAGE_INTERPOLATE_KERNEL_H__

__global__ void
interpolate_kernel(const uchar4* src0, const uchar4* src1, uchar4* dst,
		   uint width, uint height, float ratio)
{
    const uint		x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint		y = blockIdx.y * blockDim.y + threadIdx.y;
    
    const uint		n = x + y * width;

    const float		r1 = ratio;
    const float		r0 = 1.0f - r1;
    
    const uchar4	s0 = src0[n];
    const uchar4	s1 = src1[n];

    dst[n].x = s0.x * r0 + s1.x * r1;
    dst[n].y = s0.y * r0 + s1.y * r1;
    dst[n].z = s0.z * r0 + s1.z * r1;
}

#endif // #ifndef __CUDA_IMAGE_INTERPOLATE_KERNEL_CU__
