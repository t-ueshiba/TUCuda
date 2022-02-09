// Software License Agreement (BSD License)
//
// Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of National Institute of Advanced Industrial
//    Science and Technology (AIST) nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Toshio Ueshiba
//
/*!
  \file		Texture.h
  \brief	CUDAテクスチャメモリに関連するクラスの定義と実装
*/
#pragma once

#include "TU/cu/algorithm.h"
#include "TU/cu/Array++.h"
#include <cuda_texture_types.h>

namespace TU
{
namespace cu
{
/************************************************************************
*  class Texture<T>							*
************************************************************************/
//! CUDAにおけるT型オブジェクトのテクスチャクラス
/*!
  既存の1次元または2次元のCUDA配列とファイルスコープで定義された
  テクスチャ参照から生成される．
  \param T	要素の型
*/
template <class T>
class Texture
{
  public:
    using value_type	= T;

  public:
    Texture(const Array<T>& a, bool normalize=false, bool interpolate=true,
	    enum cudaTextureAddressMode addressMode=cudaAddressModeBorder);
    Texture(const Array2<T>& a, bool normalize=false, bool interpolate=true,
	    enum cudaTextureAddressMode addressMode=cudaAddressModeBorder);
    ~Texture()							;

    cudaTextureObject_t	get()				const	{ return _tex; }

  private:
    cudaTextureObject_t	_tex;
};

//! 1次元CUDA配列から1次元テクスチャを作る．
/*!
  \param a		1次元CUDA配列
  \param normalize	テクスチャを読み出すときに[0, 1](符号なし)または
			[-1, 1](符号付き)に正規化するならtrue,
			そうでなければfalse
  \param interpolate	テクスチャを読み出すときに補間を行うならtrue,
			そうでなければfalse
  \param addressMode
*/
template <class T>
Texture<T>::Texture(const Array<T>& a, bool normalize, bool interpolate,
		    enum cudaTextureAddressMode addressMode)
    :_tex(0)
{
    cudaResourceDesc	resdesc;
    memset(&resdesc, 0, sizeof(resdesc));
    resdesc.resType		   = cudaResourceTypeLinear;
    resdesc.res.linear.devPtr	   = a.data().get();
    resdesc.res.linear.desc	   = cudaCreateChannelDesc<T>();
    resdesc.res.linear.sizeInBytes = a.size()*sizeof(T);

    cudaTextureDesc	texdesc;
    memset(&texdesc, 0, sizeof(texdesc));
    texdesc.addressMode[0]   = addressMode;
    texdesc.filterMode	     = (interpolate ? cudaFilterModeLinear
					    : cudaFilterModePoint);
    texdesc.readMode	     = (normalize   ? cudaReadModeNormalizedFloat
					    : cudaReadModeElementType);
    texdesc.normalizedCoords = 0;

    const auto	err = cudaCreateTextureObject(&_tex, &resdesc, &texdesc, NULL);
    if (err != cudaSuccess)
	throw std::runtime_error("Texture<T>::Texture(): failed to create a texture object bound to the given 1D array!");
}

//! 2次元CUDA配列から2次元テクスチャを作る．
/*!
  \param a		2次元CUDA配列
  \param normalize	テクスチャを読み出すときに[0, 1](符号なし)または
			[-1, 1](符号付き)に正規化するならtrue,
			そうでなければfalse
  \param interpolate	テクスチャを読み出すときに補間を行うならtrue,
			そうでなければfalse
  \param addressMode
*/
template <class T>
Texture<T>::Texture(const Array2<T>& a, bool normalize, bool interpolate,
		    enum cudaTextureAddressMode addressMode)
    :_tex(0)
{
    cudaResourceDesc	resdesc;
    memset(&resdesc, 0, sizeof(resdesc));
    resdesc.resType			= cudaResourceTypePitch2D;
    resdesc.res.pitch2D.devPtr		= a.data().get();
    resdesc.res.pitch2D.desc		= cudaCreateChannelDesc<T>();
    resdesc.res.pitch2D.width		= a.ncol();
    resdesc.res.pitch2D.height		= a.nrow();
    resdesc.res.pitch2D.pitchInBytes	= a.stride()*sizeof(T);

    cudaTextureDesc	texdesc;
    memset(&texdesc, 0, sizeof(texdesc));
    texdesc.addressMode[0]   = addressMode;
    texdesc.addressMode[1]   = addressMode;
    texdesc.filterMode	     = (interpolate ? cudaFilterModeLinear
					    : cudaFilterModePoint);
    texdesc.readMode	     = (normalize   ? cudaReadModeNormalizedFloat
					    : cudaReadModeElementType);
    texdesc.normalizedCoords = 0;

    const auto	err = cudaCreateTextureObject(&_tex, &resdesc, &texdesc, NULL);
    if (err != cudaSuccess)
	throw std::runtime_error("Texture<T>::Texture(): failed to create a texture object bound to the given 2D array!");
}

//! テクスチャを破壊する．
template <class T> inline
Texture<T>::~Texture()
{
    cudaDestroyTextureObject(_tex);
}

/************************************************************************
*  warp(const Array2<T>& a, OUT out, OP op)				*
************************************************************************/
//! CUDAによって2次元配列の転置処理を行う．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class BOX_TRAITS=BlockTraits<>, class T, class OUT, class MAP> void
warp(const Array2<T>& a, OUT out, MAP map)				;

#if defined(__NVCC__)
namespace device
{
  template <class T, class OUT, class MAP> __global__ void
  warp(cudaTextureObject_t tex, range<range_iterator<OUT> > out, MAP map)
  {
      const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (y < out.size() && x < out.begin().size())
      {
	  const auto	p = map(x, y);

	  out[y][x] = tex2D<T>(tex, p.x, p.y);
      }
  }
}	// namespace device

template <class BLOCK_TRAITS, class T, class OUT, class MAP> inline void
warp(const Array2<T>& a, OUT out, MAP map)
{
    const int	nrow = a.nrow();
    if (nrow < 1)
	return;

    const int	ncol = a.ncol();
    if (ncol < 1)
	return;

    const Texture<T>	tex(a);

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol, threads.x), divUp(nrow, threads.y));
    device::warp<T><<<blocks, threads>>>(tex.get(),
					 cu::make_range(out, nrow), map);
}
#endif

}	// namespace cu
}	// namespace TU
