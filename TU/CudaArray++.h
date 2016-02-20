/*
 *  $Id$
 */
/*!
  \mainpage	libTUCuda++ - NVIDIA社のCUDAを利用するためのユティリティライブラリ
  \anchor	libTUCuda

  \section copyright 著作権
  平成14-23年（独）産業技術総合研究所 著作権所有

  創作者：植芝俊夫

  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
  等の行為を禁止します．
   
  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
  よび創作者は責任を負いません。

  Copyright 2002-2011.
  National Institute of Advanced Industrial Science and Technology (AIST)

  Creator: Toshio UESHIBA

  [AIST Confidential and all rights reserved.]
  This program is confidential. Any using, copying, changing or
  giving any information concerning with this program to others
  without permission by the copyright holder are strictly prohibited.

  [No Warranty.]
  The copyright holder or the creator are not responsible for any
  damages caused by using this program.

  \section abstract 概要
  libTUCuda++は，C++環境においてNVIDIA社のCUDAを利用するためのユティリティ
  ライブラリである．以下のようなクラスおよび関数が実装されている．

  <b>デバイス側のグローバルメモリ領域にとられる1次元および2次元配列</b>
  - #TU::CudaArray
  - #TU::CudaArray2

  <b>デバイス側のテクスチャメモリ</b>
  - #TU::CudaTexture
  
  <b>フィルタリング</b>
  - #TU::CudaFilter2
  - #TU::CudaGaussianConvolver2

  <b>ユティリティ</b>
  - #TU::cudaCopyToConstantMemory(Iterator, Iterator, T*)
  - #TU::cudaSubsample(const CudaArray2<T>&, CudaArray2<T>&)
  - #TU::cudaOp3x3(const CudaArray2<S>&, CudaArray2<T>&, OP op)
  - #TU::cudaSuppressNonExtrema3x3(const CudaArray2<T>&, CudaArray2<T>&, OP op, T)
  
  \file		CudaArray++.h
  \brief	CUDAデバイス上の配列に関連するクラスの定義と実装
*/
#ifndef __TU_CUDAARRAYPP_H
#define __TU_CUDAARRAYPP_H

#include <thrust/device_allocator.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include "TU/Array++.h"

/*!
  \namespace	TU
  \brief	本ライブラリで定義されたクラスおよび関数を納める名前空間
*/
namespace TU
{
/************************************************************************
*  specialization for Buf<T, 0, thrust::device_allocator<T> >		*
************************************************************************/
  /*
template <class T> inline
Buf<T, 0, thrust::device_allocator<T> >::Buf(const Buf& b)
    :_size(b._size), _p(alloc(_size)), _shared(0), _capacity(_size)
{
    thrust::copy(b.cbegin(), b.cend(), begin());
}
    
template <class T> inline Buf<T, 0, thrust::device_allocator<T> >&
Buf<T, 0, thrust::device_allocator<T> >::operator =(const Buf& b)
{
    if (this != &b)
    {
	resize(b._size);
	thrust::copy(b.cbegin(), b.cend(), begin());
    }
    return *this;
}
  */
/************************************************************************
*  class CudaBuf<T>							*
************************************************************************/
//! CUDAにおいてデバイス側に確保される可変長バッファクラス
/*!
  単独で使用することはなく，#TU::Arrayまたは#TU::Array2の
  第2テンプレート引数に指定することによって，それらの基底クラスとして使う．
  \param T	要素の型
*/
template <class T>
class Buf<T, 0, thrust::device_allocator<T> >
{
  public:
  //! アロケータの型    
    typedef thrust::device_allocator<T>			allocator_type;
  //! 要素の型    
    typedef typename allocator_type::value_type		value_type;
  //! 要素への参照    
    typedef typename allocator_type::reference		reference;
  //! 定数要素への参照    
    typedef typename allocator_type::const_reference	const_reference;
  //! 要素へのポインタ    
    typedef typename allocator_type::pointer		pointer;
  //! 定数要素へのポインタ    
    typedef typename allocator_type::const_pointer	const_pointer;
  //! 反復子    
    typedef pointer					iterator;
  //! 定数反復子    
    typedef const_pointer				const_iterator;
    
  public:
    explicit		Buf(size_t siz=0)
			    :_size(siz),
			     _p(alloc(_size)), _shared(false)		{}
			Buf(pointer p, size_t siz)
			    :_size(siz), _p(p), _shared(true)		{}
			Buf(const Buf& b)
			    :_size(b._size), _p(alloc(_size)), _shared(false)
			{
			    thrust::copy(b.cbegin(), b.cend(), begin());
			}
			
    Buf&		operator =(const Buf& b)
			{
			    if (this != &b)
			    {
				resize(b._size);
				thrust::copy(b.cbegin(), b.cend(), begin());
			    }
			    return *this;
			}

			~Buf()
			{
			    if (!_shared)
				free(_p, _size);
			}

    pointer		data()				{ return _p; }
    const_pointer	data()			const	{ return _p; }
    iterator		begin()				{ return _p; }
    const_iterator	begin()			const	{ return _p; }
    const_iterator	cbegin()		const	{ return _p; }
    iterator		end()				{ return _p + _size; }
    const_iterator	end()			const	{ return _p + _size; }
    const_iterator	cend()			const	{ return _p + _size; }
    size_t		size()			const	{ return _size; }
    
  //! バッファの要素数を変更する．
  /*!
    ただし，他のオブジェクトと記憶領域を共有しているバッファの要素数を
    変更することはできない．
    \param siz			新しい要素数
    \return			sizが元の要素数よりも大きければtrue，そう
				でなければfalse
    \throw std::logic_error	記憶領域を他のオブジェクトと共有している場合
				に送出
  */
    bool		resize(size_t siz)
			{
			    if (_size == siz)
				return false;
    
			    if (_shared)
				throw std::logic_error("Buf<T>::resize: cannot change size of shared buffer!");
			    
			    const size_t	old_size = _size;
			    _size = siz;
			    free(_p, old_size);
			    _p = alloc(_size);

			    return _size > old_size;
			}

  //! バッファが内部で使用する記憶領域を指定したものに変更する．
  /*!
    \param p	新しい記憶領域へのポインタ
    \param siz	新しい要素数
  */
    void		resize(pointer p, size_t siz)
			{
			    if (!_shared)
				free(_p, _size);
			    _size = siz;
			    _p = p;
			    _shared = true;
			}
	
    std::istream&	get(std::istream& in, size_t m=0)	;

  //! 出力ストリームに配列を書き出す(ASCII)．
  /*!
    \param out	出力ストリーム
    \return	outで指定した出力ストリーム
  */
    std::ostream&	put(std::ostream& out) const
			{
			    for (size_t i = 0; i < _size; )
				out << ' ' << _p[i++];
			    return out;
			}
    
  private:
    pointer		alloc(size_t siz)
			{
			    return _allocator.allocate(siz);
			}

    void		free(pointer p, size_t siz)
			{
			    _allocator.deallocate(p, siz);
			}

  private:
    allocator_type	_allocator;
    size_t		_size;		// the number of elements in the buffer
    pointer		_p;		// pointer to the buffer area
    bool		_shared;	// whether shared with other objects
};

//! 入力ストリームから指定した箇所に配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param m	読み込み先の先頭を指定するindex
  \return	inで指定した入力ストリーム
*/
template <class T> std::istream&
Buf<T, 0, thrust::device_allocator<T> >::get(std::istream& in, size_t m)
{
    const size_t	BufSiz = (sizeof(T) < 2048 ? 2048 / sizeof(T) : 1);
    T* const		tmp = new T[BufSiz];
    size_t		n = 0;
    
    while (n < BufSiz)
    {
	char	c;
	while (in.get(c))		// skip white spaces other than '\n'
	    if (!isspace(c) || c == '\n')
		break;

	if (!in || c == '\n')
	{
	    resize(m + n);
	    break;
	}

	in.putback(c);
	in >> tmp[n++];
    }
    if (n == BufSiz)
	get(in, m + n);

    for (size_t i = 0; i < n; ++i)
	_p[m + i] = tmp[i];

    delete [] tmp;
    
    return in;
}
    
/************************************************************************
*  class CudaArray<T>							*
************************************************************************/
//! CUDAにおいてデバイス側に確保されるT型オブジェクトの1次元配列クラス
/*!
  \param T	要素の型
*/
template <class T>
class CudaArray : public Array<T, 0, thrust::device_allocator<T> >
{
  private:
    typedef Array<T, 0, thrust::device_allocator<T> >	super;

  public:
  //! 成分の型
    typedef typename super::element_type	element_type;
  //! 要素の型    
    typedef typename super::value_type		value_type;
  //! 要素への参照
    typedef typename super::reference		reference;
  //! 定数要素への参照
    typedef typename super::const_reference	const_reference;
  //! 要素へのポインタ
    typedef typename super::pointer		pointer;
  //! 定数要素へのポインタ
    typedef typename super::const_pointer	const_pointer;
  //! 反復子
    typedef typename super::iterator		iterator;
  //! 定数反復子
    typedef typename super::const_iterator	const_iterator;
  //! 逆反復子    
    typedef typename super::reverse_iterator	reverse_iterator;
  //! 定数逆反復子    
    typedef typename super::const_reverse_iterator
						const_reverse_iterator;
  //! ポインタ間の差
    typedef typename super::difference_type	difference_type;
  //! 要素への直接ポインタ
    typedef element_type*			raw_pointer;
  //! 定数要素への直接ポインタ
    typedef const element_type*			const_raw_pointer;
    
  public:
		CudaArray()			:super()		{}
    explicit	CudaArray(size_t d)		:super(d)		{}
		CudaArray(pointer p, size_t d)	:super(p, d)		{}
		CudaArray(CudaArray& a, size_t i, size_t d)
		    :super(a, i, d)					{}
    template <size_t D>
		CudaArray(const Array<T, D>& a)	:super(a.size())
		{
		    thrust::copy(a.begin(), a.end(), begin());
		}
    template <size_t D>
    CudaArray&	operator =(const Array<T, D>& a)
		{
		    resize(a.size());
		    thrust::copy(a.begin(), a.end(), begin());
		    return *this;
		}
    CudaArray&	operator =(const element_type& c)
		{
		    thrust::fill(begin(), end(), c);
		    return *this;
		}
    template <size_t D> const CudaArray&
		write(Array<T, D>& a) const
		{
		    a.resize(size());
		    thrust::copy(begin(), end(), a.begin());
		    return *this;
		}

    raw_pointer		data()			{ return super::data().get(); }
    const_raw_pointer	data()		const	{ return super::data().get(); }
    
    using	super::begin;
    using	super::end;
    using	super::rbegin;
    using	super::rend;
    using	super::size;
    using	super::dim;
    using	super::resize;
};

/************************************************************************
*  class CudaArray2<T>							*
************************************************************************/
//! CUDAにおいてデバイス側に確保されるT型オブジェクトの2次元配列クラス
/*!
  \param T	要素の型
*/
template <class T>
class CudaArray2 : public Array2<CudaArray<T> >
{
  private:
    typedef Buf<T, 0, thrust::device_allocator<T> >	buf_type;
    typedef Array2<CudaArray<T> >			super;
    
  public:
  //! 行の型    
    typedef typename super::value_type			value_type;
  //! 行への参照    
    typedef typename super::reference			reference;
  //! 定数行への参照    
    typedef typename super::const_reference		const_reference;
  //! 行の反復子    
    typedef typename super::iterator			iterator;
  //! 行の定数反復子    
    typedef typename super::const_iterator		const_iterator;
  //! 行の逆反復子    
    typedef typename super::reverse_iterator		reverse_iterator;
  //! 行の定数逆反復子    
    typedef typename super::const_reverse_iterator	const_reverse_iterator;
  //! 成分の型    
    typedef typename super::element_type		element_type;
  //! 成分へのポインタ    
    typedef typename super::pointer			pointer;
  //! 定数成分へのポインタ    
    typedef typename super::const_pointer		const_pointer;
  //! 成分への直接ポインタ    
    typedef element_type*				raw_pointer;
  //! 定数成分への直接ポインタ    
    typedef const element_type*				const_raw_pointer;
  //! ポインタ間の差    
    typedef std::ptrdiff_t				difference_type;

  public:
		CudaArray2()			:super()		{}
		CudaArray2(size_t r, size_t c)	:super(r, c)		{}
		CudaArray2(pointer p, size_t r, size_t c)
		    :super(p, r, c)					{}
		CudaArray2(CudaArray2& a,
			   size_t i, size_t j, size_t r, size_t c)
		    :super(a, i, j, r, c)				{}
    template <class T2, size_t R2, size_t C2>
		CudaArray2(const Array2<T2, R2, C2>& a)
		    :super()
		{
		    operator =(a);
		}
    template <class T2, size_t R2, size_t C2>
    CudaArray2&	operator =(const Array2<T2, R2, C2>& a)
		{
		    resize(a.nrow(), a.ncol());
		    if (a.nrow() > 0 && a.stride() == stride())
		    {
			thrust::copy(a[0].begin(), a[a.nrow()-1].end(),
				     (*this)[0].begin());
		    }
		    else
		    {
			for (size_t i = 0; i < nrow(); ++i)
			    (*this)[i] = a[i];
		    }
		    return *this;
		}

    CudaArray2&	operator =(const element_type& c)
		{
		    if (nrow() > 0)
			thrust::fill((*this)[0].begin(),
				     (*this)[nrow()-1].end(), c);
		    return *this;
		}
    
    template <class T2, size_t R2, size_t C2> const CudaArray2&
		write(Array2<T2, R2, C2>& a) const
		{
		    a.resize(nrow(), ncol());
		    if (nrow() > 0 && stride() == a.stride())
		    {
			thrust::copy((*this)[0].begin(),
				     (*this)[nrow()-1].end(), a[0].begin());
		    }
		    else
		    {
			for (size_t i = 0; i < nrow(); ++i)
			    (*this)[i].write(a[i]);
		    }
		    return *this;
		}
	
    raw_pointer		data()		{ return super::data().get(); }
    const_raw_pointer	data()	const	{ return super::data().get(); }
    
    using	super::begin;
    using	super::end;
    using	super::size;
    using	super::dim;
    using	super::nrow;
    using	super::ncol;
    using	super::stride;
};

}
#endif	// !__TU_CUDAARRAYPP_H
