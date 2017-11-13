/*
 *  $Id$
 */
/*!
  \file		Texture.h
  \brief	CUDAテクスチャメモリに関連するクラスの定義と実装
*/ 
#ifndef TU_CUDA_TEXTURE_H
#define TU_CUDA_TEXTURE_H

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
    using array_type	= Array<T>;
    using array2_type	= Array2<T>;
    using texref_type	= textureReference;
    
  public:
    template <enum cudaTextureReadMode M>
    Texture(const array_type& a, texture<T, 1, M>& texref,
	    bool wrap=false,
	    bool interpolate=false, bool normalized=false)		;
    template <enum cudaTextureReadMode M>
    Texture(const array2_type& a, texture<T, 2, M>& texref,
	    bool wrap=false,
	    bool interpolate=false, bool normalized=false)		;
    ~Texture()								;
    
  private:
    texref_type&	_texref;
};

//! 1次元CUDA配列から1次元テクスチャを作る．
/*!
  \param a		1次元CUDA配列
  \param texref		ファイルスコープで定義されたテクスチャ参照
  \param wrap		範囲外をアクセスしたときにラッピングするならtrue,
			そうでなければfalse
  \param interpolate	テクスチャを読み出すときに補間を行うならtrue,
			そうでなければfalse
  \param normalized	テクスチャを読み出すときに[0, 1](符号なし)または
			[-1, 1](符号付き)に正規化するならtrue,
			そうでなければfalse
*/
template <class T> template <enum cudaTextureReadMode M> inline
Texture<T>::Texture(const array_type& a, texture<T, 1, M>& texref,
		    bool wrap, bool interpolate, bool normalized)
    :_texref(texref)
{
    _texref.addressMode[0] = (wrap ? cudaAddressModeWrap
				   : cudaAddressModeClamp);
    _texref.filterMode	   = (interpolate ? cudaFilterModeLinear
					  : cudaFilterModePoint);
    _texref.normalized	   = normalized;
    
    const auto	err = cudaBindTexture(0, &_texref, a.data(),
				      &_texref.channelDesc, a.size());
    if (err != cudaSuccess)
	throw std::runtime_error("Texture::Texture(): failed to bind texture to the given 1D array!");
}

//! 2次元CUDA配列から2次元テクスチャを作る．
/*!
  \param a		2次元CUDA配列
  \param texref		ファイルスコープで定義されたテクスチャ参照
  \param wrap		範囲外をアクセスしたときにラッピングするならtrue,
			そうでなければfalse
  \param interpolate	テクスチャを読み出すときに補間を行うならtrue,
			そうでなければfalse
  \param normalized	テクスチャを読み出すときに[0, 1](符号なし)または
			[-1, 1](符号付き)に正規化するならtrue,
			そうでなければfalse
*/
template <class T> template <enum cudaTextureReadMode M> inline
Texture<T>::Texture(const array2_type& a, texture<T, 2, M>& texref,
		    bool wrap, bool interpolate, bool normalized)
    :_texref(texref)
{
    _texref.addressMode[0] = (wrap ? cudaAddressModeWrap
				   : cudaAddressModeClamp);
    _texref.addressMode[1] = _texref.addressMode[0];
    _texref.filterMode	   = (interpolate ? cudaFilterModeLinear
					  : cudaFilterModePoint);
    _texref.normalized	   = normalized;
    
    const auto	err = cudaBindTexture2D(0, &_texref, a.data(),
					&_texref.channelDesc,
					a.ncol(), a.nrow(),
					a.stride()*sizeof(T));
    if (err != cudaSuccess)
	throw std::runtime_error("Texture::Texture(): failed to bind texture to the given 2D array!");
}

//! テクスチャを破壊する．
template <class T> inline
Texture<T>::~Texture()
{
    cudaUnbindTexture(&_texref);
}

}
}
#endif	// !TU_CUDA_TEXTURE_H
