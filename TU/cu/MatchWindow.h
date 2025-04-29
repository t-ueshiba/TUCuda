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

namespace TU::cu
{
#if defined(__NVCC__)
/************************************************************************
*  match_window()							*
************************************************************************/
namespace device
{
  template <size_t BLOCK_DIM=16, size_t WIN_SIZE_MAX=32,
	    class IN, class OUT, class OP>
  __global__ void
  match_window(range<range_iterator<IN> > in, range<range_iterator<IN> > win,
	       range<range_iterator<OUT> > out, OP diff)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      const int	winSizeX = win.begin().size();
      const int	winSizeY = win.size();
      const int	x0 = __mul24(blockIdx.x, blockDim.x);
      const int	y0 = __mul24(blockIdx.y, blockDim.y);
      const int	xsiz = ::min(blockDim.x + winSizeX - 1, in.begin().size() - x0);
      const int	ysiz = ::min(blockDim.y + winSizeY - 1, in.size() - y0);

      __shared__ value_type	in_s[BLOCK_DIM + WIN_SIZE_MAX - 1]
				    [BLOCK_DIM + WIN_SIZE_MAX];
      loadTile(slice(in.cbegin(), y0, ysiz, x0, xsiz), in_s);
      __syncthreads();

      if (v >= in.size() || u >= in.cbegin().size())
	  return;

      value_type	wd_sum{0};
      T			w_sum{0};

      for (int y = 0; y < winSizeY; ++y)
	  for (int x = 0; x < winSizeX; ++x)
	  {
	      const auto	d = win[y][x]
				  - in_s[y + threadIdx.y][x + threadIdx.x];
	      wd_sum += weight*dd;
	      w_sum  += weight;
	  }

      out[y0 + threadIdx.y][x0 + threadIdx.x] = wd_sum/w_sum;
  }
}

template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT, class T> void
match_window(IN in, IN ie, OUT out,
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

    device::match_window<<<blocks, threads>>>(
	cu::make_range(in,  nrow), cu::make_range(out, nrow), kernel_size,
	0.5/(sigma_spatial*sigma_spatial),
	0.5/(sigma_depth*sigma_depth));
    gpuCheckLastError();
};
#endif	// __NVCC__
}	// namespace TU::cu

