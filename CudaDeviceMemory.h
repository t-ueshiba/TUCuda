/*
 *  平成14-21年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2009.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id: CudaDeviceMemory.h,v 1.7 2009-08-13 23:00:37 ueshiba Exp $
 */
#ifndef __TUCudaDeviceMemory_h
#define __TUCudaDeviceMemory_h

#include <cutil.h>
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class CudaBuf<T>							*
************************************************************************/
//! CUDAにおいてデバイス側に確保される可変長バッファクラス
/*!
  単独で使用することはなく，#TU::CudaDeviceMemoryまたは#TU::Array2の
  第2テンプレート引数に指定することによって，それらの基底クラスとして使う．
  外部に確保した記憶領域を割り当てることはできない．
  \param T	要素の型
*/
template <class T>
class CudaBuf : private Buf<T>
{
  public:
    explicit CudaBuf(u_int siz=0)				;
    CudaBuf(const CudaBuf& b)					;
    CudaBuf&		operator =(const CudaBuf& b)		;
    ~CudaBuf()							;

    using		Buf<T>::operator T*;
    using		Buf<T>::operator const T*;
    using		Buf<T>::size;

    bool		resize(u_int siz)			;
    static u_int	align(u_int siz)			;
    
  private:
    static T*		memalloc(u_int siz)			;
    static void		memfree(T* p)				;

    enum		{ALIGN = 32};	//!< thread数/warp
};

//! 指定した要素数のバッファを生成する．
/*!
  \param siz	要素数
*/
template <class T> inline
CudaBuf<T>::CudaBuf(u_int siz)
    :Buf<T>(memalloc(siz), siz)
{
}

//! コピーコンストラクタ
template <class T> inline
CudaBuf<T>::CudaBuf(const CudaBuf& b)
    :Buf<T>(memalloc(b.size()), b.size())
{
    CUDA_SAFE_CALL(cudaMemcpy((T*)*this, (const T*)b,
			      size()*sizeof(T), cudaMemcpyDeviceToDevice));
}
    
//! 標準代入演算子
template <class T> inline CudaBuf<T>&
CudaBuf<T>::operator =(const CudaBuf& b)
{
    resize(b.size());
    CUDA_SAFE_CALL(cudaMemcpy((T*)*this, (const T*)b,
			      size()*sizeof(T), cudaMemcpyDeviceToDevice));
    return *this;
}
    
//! デストラクタ
template <class T> inline
CudaBuf<T>::~CudaBuf()
{
    memfree((T*)*this);
}
    
//! バッファの要素数を変更する．
/*!
  \param siz			新しい要素数
  \return			sizが元の要素数と等しければtrue，そうでなければ
				false
*/
template <class T> inline bool
CudaBuf<T>::resize(u_int siz)
{
    if (siz == size())
	return false;

    memfree((T*)*this);
    Buf<T>::resize(memalloc(siz), siz);
    return true;
}

//! 指定された要素数を持つ記憶領域を確保するために実際に必要な要素数を返す．
/*!
  （記憶容量ではなく）要素数が32の倍数になるよう，与えられた要素数を繰り上げる．
  \param siz	要素数
  \return	32の倍数に繰り上げられた要素数
*/
template <class T> inline u_int
CudaBuf<T>::align(u_int siz)
{
    return (siz > 0 ? ALIGN * ((siz - 1) / ALIGN + 1) : 0);
}

template <class T> inline T*
CudaBuf<T>::memalloc(u_int siz)
{
    using namespace	std;
    
    T*	p = 0;
    if (siz > 0)
    {
	CUDA_SAFE_CALL(cudaMalloc((void**)&p, align(siz)*sizeof(T)));
	if (p == 0)
	    throw runtime_error("Failed to allocate CUDA device memory!!");
    }
    return p;
}

template <class T> inline void
CudaBuf<T>::memfree(T* p)
{
    if (p != 0)
	CUDA_SAFE_CALL(cudaFree(p));
}

/************************************************************************
*  class CudaDeviceMemory<T, B>						*
************************************************************************/
//! CUDAにおいてデバイス側に確保されるT型オブジェクトの1次元メモリ領域クラス
/*!
  \param T	要素の型
  \param B	バッファ
*/
template <class T, class B=CudaBuf<T> >
class CudaDeviceMemory : private Array<T, B>
{
  public:
    typedef T			value_type;	  //!< 要素の型
    typedef ptrdiff_t		difference_type;  //!< ポインタ間の差
    typedef value_type*		pointer;	  //!< 要素へのポインタ
    typedef const value_type*	const_pointer;	  //!< 定数要素へのポインタ
    
  public:
    CudaDeviceMemory()						;
    explicit CudaDeviceMemory(u_int d)				;

    using	Array<T, B>::operator pointer;
    using	Array<T, B>::operator const_pointer;
    using	Array<T, B>::size;
    using	Array<T, B>::dim;
    using	Array<T, B>::resize;
    
    template <class T2, class B2> CudaDeviceMemory&
		readFrom(const Array<T2, B2>& a)		;
    template <class T2, class B2> const CudaDeviceMemory&
		writeTo(Array<T2, B2>& a)		const	;
};

//! CUDAデバイスメモリ領域を生成する．
template <class T, class B>
CudaDeviceMemory<T, B>::CudaDeviceMemory()
    :Array<T, B>()
{
}

//! 指定した要素数のCUDAデバイスメモリ領域を生成する．
/*!
  \param d	メモリ領域の要素数
*/
template <class T, class B>
CudaDeviceMemory<T, B>::CudaDeviceMemory(u_int d)
    :Array<T, B>(d)
{
}

//! ホスト側の配列をこのデバイス側メモリ領域に読み込む．
/*!
  \param a	コピー元の配列
  \return	このメモリ領域
*/
template <class T, class B>
template <class T2, class B2> inline CudaDeviceMemory<T, B>&
CudaDeviceMemory<T, B>::readFrom(const Array<T2, B2>& a)
{
    using namespace	std;
    
    if (sizeof(T) != sizeof(T2))
	throw logic_error(
	    "CudaDeviceMemory<T, B>::readFrom: mismatched element sizes!!");
    resize(a.dim());
    CUDA_SAFE_CALL(cudaMemcpy(pointer(*this), (const T2*)a, 
			      dim() * sizeof(T), cudaMemcpyHostToDevice));
    return *this;
}
    
//! このデバイス側メモリ領域の内容を領域ホスト側の配列に書き出す．
/*!
  \param a	コピー先の配列
  \return	このメモリ領域
*/
template <class T, class B>
template <class T2, class B2> inline const CudaDeviceMemory<T, B>&
CudaDeviceMemory<T, B>::writeTo(Array<T2, B2>& a) const
{
    using namespace	std;
    
    if (sizeof(T) != sizeof(T2))
	throw logic_error(
	    "CudaDeviceMemory<T, B>::writeTo: mismatched element sizes!!");
    a.resize(dim());
    CUDA_SAFE_CALL(cudaMemcpy((T2*)a, const_pointer(*this),
			      dim() * sizeof(T), cudaMemcpyDeviceToHost));
    return *this;
}

/************************************************************************
*  class CudaDeviceMemory2<T, R>					*
************************************************************************/
//! CUDAにおいてデバイス側に確保されるT型オブジェクトの2次元メモリ領域クラス
/*!
  \param T	要素の型
  \param R	行バッファ
*/
template <class T, class R=Buf<CudaDeviceMemory<T, Buf<T> > > >
class CudaDeviceMemory2
    : private Array2<CudaDeviceMemory<T, Buf<T> >, CudaBuf<T>, R>
{
  private:
    typedef Array2<CudaDeviceMemory<T, Buf<T> >, CudaBuf<T>, R>	super;
    
  public:
    typedef CudaDeviceMemory<T,Buf<T> >	row_type;	//!< 行の型
    typedef R				rowbuffer_type;	//!< 行バッファの型
    typedef CudaBuf<T>			buffer_type;	//!< バッファの型
    typedef T				value_type;	//!< 要素の型
    typedef ptrdiff_t			difference_type;//!< ポインタ間の差
    typedef value_type*			pointer;	//!< 要素へのポインタ
    typedef const value_type*		const_pointer;	//!< 定数要素へのポインタ

  public:
    CudaDeviceMemory2()							;
    CudaDeviceMemory2(u_int r, u_int c)					;
    CudaDeviceMemory2(const CudaDeviceMemory2& m)			;
    CudaDeviceMemory2&	operator =(const CudaDeviceMemory2& m)		;
    
    using	super::operator pointer;
    using	super::operator const_pointer;
    using	super::operator [];
    using	super::begin;
    using	super::end;
    using	super::size;
    using	super::dim;
    using	super::nrow;
    using	super::ncol;
    using	super::resize;
    
    template <class T2, class B2, class R2> CudaDeviceMemory2&
		readFrom(const Array2<T2, B2, R2>& a)			;
    template <class T2, class B2, class R2> const CudaDeviceMemory2&
		writeTo(Array2<T2, B2, R2>& a)			const	;
};

//! 2次元CUDAデバイスメモリ領域を生成する．
template <class T, class R> inline
CudaDeviceMemory2<T, R>::CudaDeviceMemory2()
    :super()
{
}

//! 行数と列数を指定して2次元CUDAデバイスメモリ領域を生成する．
/*!
  \param r	行数
  \param c	列数
*/
template <class T, class R> inline
CudaDeviceMemory2<T, R>::CudaDeviceMemory2(u_int r, u_int c)
    :super(r, c)
{
}

//! コピーコンストラクタ
template <class T, class R> inline
CudaDeviceMemory2<T, R>::CudaDeviceMemory2(const CudaDeviceMemory2& m)
    :super(m.nrow(), m.ncol())
{
    if (nrow() > 1)
    {
	const u_int	stride = pointer((*this)[1]) - pointer((*this)[0]);
	if (const_pointer(m[1]) - const_pointer(m[0]) == stride)
	{
	    CUDA_SAFE_CALL(cudaMemcpy(pointer(*this), const_pointer(m),
				      nrow()*stride*sizeof(T),
				      cudaMemcpyDeviceToDevice));
	    return;
	}
    }
    super::operator =(m);
}
    
//! 標準代入演算子
template <class T, class R> inline CudaDeviceMemory2<T, R>&
CudaDeviceMemory2<T, R>::operator =(const CudaDeviceMemory2& m)
{
    if (this != &m)
    {
	resize(m.nrow(), m.ncol());
	if (nrow() > 1)
	{
	    const u_int	stride = pointer((*this)[1]) - pointer((*this)[0]);
	    if (const_pointer(m[1]) - const_pointer(m[0]) == stride)
	    {
		CUDA_SAFE_CALL(cudaMemcpy(pointer(*this), const_pointer(m),
					  nrow()*stride*sizeof(T),
					  cudaMemcpyDeviceToDevice));
		return *this;
	    }
	}
	super::operator =(m);
    }
    return *this;
}
    
//! ホスト側の配列をこのデバイス側メモリ領域に読み込む．
/*!
  \param a	コピー元の配列
  \return	このメモリ領域
*/
template <class T, class R>
template <class T2, class B2, class R2> CudaDeviceMemory2<T, R>&
CudaDeviceMemory2<T, R>::readFrom(const Array2<T2, B2, R2>& a)
{
    typedef typename Array2<T2, B2, R2>::const_pointer	const_pointer2;
    
    resize(a.nrow(), a.ncol());
    if (nrow() > 1)
    {
	const u_int	stride = pointer((*this)[1]) - pointer((*this)[0]);
	if (const_pointer2(a[1]) - const_pointer2(a[0]) == stride)
	{
	    CUDA_SAFE_CALL(cudaMemcpy(pointer(*this), const_pointer2(a),
				      nrow()*stride*sizeof(T),
				      cudaMemcpyHostToDevice));
	    return *this;
	}
    }
    for (u_int i = 0; i < nrow(); ++i)
	(*this)[i].readFrom(a[i]);
    return *this;
}

//! このデバイス側メモリ領域の内容を領域ホスト側の配列に書き出す．
/*!
  \param a	コピー先の配列
  \return	このメモリ領域
*/
template <class T, class R>
template <class T2, class B2, class R2> const CudaDeviceMemory2<T, R>&
CudaDeviceMemory2<T, R>::writeTo(Array2<T2, B2, R2>& a) const
{
    typedef typename Array2<T2, B2, R2>::pointer	pointer2;
    
    a.resize(nrow(), ncol());
    if (nrow() > 1)
    {
	const u_int	stride = const_pointer((*this)[1])
			       - const_pointer((*this)[0]);
	if (pointer2(a[1]) - pointer2(a[0]) == stride)
	{
	    CUDA_SAFE_CALL(cudaMemcpy(pointer2(a), const_pointer(*this),
				      nrow()*stride*sizeof(T),
				      cudaMemcpyDeviceToHost));
	    return *this;
	}
    }
    for (u_int i = 0; i < nrow(); ++i)
	(*this)[i].writeTo(a[i]);
    return *this;
}

}

#endif	/* !__TUCudaDeviceMemory_h */
