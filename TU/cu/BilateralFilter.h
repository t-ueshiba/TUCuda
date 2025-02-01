/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
#pragma once

#include "TU/cu/vec.h"
#include "TU/cu/algorithm.h"
#include "TU/cu/functional.h"

namespace TU
{
namespace cu
{
#if defined(__NVCC__)
/************************************************************************
*  bilateral_filter()							*
************************************************************************/
namespace device
{
  template <class IN, class OUT, class T> __global__ void
  bilateral_filter(range<range_iterator<IN> >  in,
		   range<range_iterator<OUT> > out,
		   int ksz, T sigma_spatial2_inv_half, T sigma_depth2_inv_half)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      const int	u = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	v = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (v >= in.size() || u >= in.cbegin().size())
	  return;

      const value_type	d  = in[v][u];
      const int		ue = ::min(u - ksz/2 + ksz, in.cbegin().size());
      const int		ve = ::min(v - ksz/2 + ksz, in.size());

      value_type	wd_sum{0};
      T			w_sum{0};

      for (int vv = ::max(v - ksz/2, 0); vv < ve; ++vv)
	  for (int uu = ::max(u - ksz/2, 0); uu < ue; ++uu)
	  {
	      const value_type	dd     = in[vv][uu];
	      const auto	sqdist = (uu - u)*(uu - u) + (vv - v)*(vv - v);
	      const auto	sqdiff = square(dd - d);
	      const auto	weight = exp(-(sqdist*sigma_spatial2_inv_half +
					       sqdiff*sigma_depth2_inv_half));
	      wd_sum += weight*dd;
	      w_sum  += weight;
	  }

      out[v][u] = wd_sum/w_sum;
  }
}

template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT, class T> void
bilateral_filter(IN in, IN ie, OUT out,
		 int kernel_size, T sigma_spatial, T sigma_depth)
{
    using std::size;

    const int	nrow = std::distance(in, ie);
    if (nrow < 1)
	return;

    const int	ncol = size(*in);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol, threads.x), divUp(nrow, threads.y));

    device::bilateral_filter<<<blocks, threads>>>(
	cu::make_range(in,  nrow), cu::make_range(out, nrow), kernel_size,
	0.5/(sigma_spatial*sigma_spatial),
	0.5/(sigma_depth*sigma_depth));
    gpuCheckLastError();
};
#endif	// __NVCC__
}	// namespace cu
}	// namespace TU
