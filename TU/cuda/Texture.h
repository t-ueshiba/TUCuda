/*
 *  $Id$
 */
/*!
  \file		Texture.h
  \brief	CUDAテクスチャメモリに関連するクラスの定義と実装
*/ 
#ifndef TU_CUDA_TEXTURE_H
#define TU_CUDA_TEXTURE_H

#include "TU/cuda/algorithm.h"
#include "TU/cuda/Array++.h"
#include <cuda_texture_types.h>

namespace TU
{
namespace cuda
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
    resdesc.res.linear.devPtr	   = TU::cuda::get(a.data());
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
    resdesc.res.pitch2D.devPtr		= TU::cuda::get(a.data());
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
template <class T, class OUT, class MAP> void
warp(const Array2<T>& a, OUT out, MAP map)				;
    
#if defined(__NVCC__)
namespace device
{
  template <class T, class OUT, class MAP, class STRIDE_O> __global__
  static void
  warp(cudaTextureObject_t tex, OUT out, MAP map,
       int x0, int y0, STRIDE_O stride_o)
  {
      const auto	x = x0 + __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const auto	y = y0 + __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
      const auto	p = map(x, y);

      advance_stride(out, y * stride_o);
      out[x] = tex2D<T>(tex, p.x, p.y);
  }
}	// namespace device

template <class T, class OUT, class MAP> inline void
warp(const Array2<T>& a, OUT out, MAP map)
{
    const auto		nrow     = a.nrow();
    const auto		ncol     = a.ncol();
    const auto		stride_o = stride(out);
    const Texture<T>	tex(a);
    
  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::warp<T><<<blocks, threads>>>(tex.get(), get(begin(*out)),
					 map, 0, 0, stride_o);
  // 右上
    const auto	x0 = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::warp<T><<<blocks, threads>>>(tex.get(), get(begin(*out)),
					 map, x0, 0, stride_o);
  // 左下
    const auto	y0 = blocks.y*threads.y;
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::warp<T><<<blocks, threads>>>(tex.get(), get(begin(*out)),
					 map, 0, y0, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::warp<T><<<blocks, threads>>>(tex.get(), get(begin(*out)),
					 map, x0, y0, stride_o);
}
#endif    

}	// namespace cuda
}	// namespace TU
#endif	// !TU_CUDA_TEXTURE_H
