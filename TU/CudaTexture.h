/*
 *  $Id$
 */
/*!
  \file		CudaTexture.h
  \brief	CUDAテクスチャメモリに関連するクラスの定義と実装
*/ 
#ifndef __TUCudaTexture_h
#define __TUCudaTexture_h

#include "TU/CudaArray++.h"
#include <cuda_texture_types.h>

namespace TU
{
/************************************************************************
*  class CudaTexture<T>							*
************************************************************************/
//! CUDAにおけるT型オブジェクトのテクスチャクラス
/*!
  既存の1次元または2次元のCUDA配列とファイルスコープで定義された
  テクスチャ参照から生成される．
  \param T	要素の型
*/
template <class T>
class CudaTexture
{
  public:
    typedef T					value_type;
    typedef CudaArray<T>			array_type;
    typedef CudaArray2<T>			array2_type;
    typedef textureReference			texref_type;
    
  public:
    template <enum cudaTextureReadMode M>
    CudaTexture(const array_type& a, texture<T, 1, M>& texref,
		bool wrap=false,
		bool interpolate=false, bool normalized=false)		;
    template <enum cudaTextureReadMode M>
    CudaTexture(const array2_type& a, texture<T, 2, M>& texref,
		bool wrap=false,
		bool interpolate=false, bool normalized=false)		;
    ~CudaTexture()							;
    
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
CudaTexture<T>::CudaTexture(const array_type& a, texture<T, 1, M>& texref,
			    bool wrap, bool interpolate, bool normalized)
    :_texref(texref)
{
    using namespace	std;

    _texref.addressMode[0] = (wrap ? cudaAddressModeWrap : cudaAddressModeClamp);
    _texref.filterMode	   = (interpolate ? cudaFilterModeLinear
					  : cudaFilterModePoint);
    _texref.normalized	   = normalized;
    
    cudaError_t	err = cudaBindTexture(0, &_texref, (const T*)a,
				      &_texref.channelDesc, a.size());
    if (err != cudaSuccess)
	throw runtime_error("CudaTexture::CudaTexture(): failed to bind texture to the given 1D array!");
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
CudaTexture<T>::CudaTexture(const array2_type& a, texture<T, 2, M>& texref,
			    bool wrap, bool interpolate, bool normalized)
    :_texref(texref)
{
    using namespace	std;
    
    _texref.addressMode[0] = (wrap ? cudaAddressModeWrap : cudaAddressModeClamp);
    _texref.addressMode[1] = _texref.addressMode[0];
    _texref.filterMode	   = (interpolate ? cudaFilterModeLinear
					  : cudaFilterModePoint);
    _texref.normalized	   = normalized;
    
    cudaError_t	err = cudaBindTexture2D(0, &_texref, (const T*)a,
					&_texref.channelDesc,
					a.ncol(), a.nrow(), a.stride()*sizeof(T));
    if (err != cudaSuccess)
	throw runtime_error("CudaTexture::CudaTexture(): failed to bind texture to the given 2D array!");
}

//! テクスチャを破壊する．
template <class T> inline
CudaTexture<T>::~CudaTexture()
{
    cudaUnbindTexture(&_texref);
}

}
#endif	/* !__TUCudaTexture_h */
