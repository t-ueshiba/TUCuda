/*
 *  $Id: CudaDeviceMemory.h,v 1.1 2009-04-15 00:32:05 ueshiba Exp $
 */
#ifndef __TUCudaDeviceMemory_h
#define __TUCudaDeviceMemory_h

#include <cutil.h>
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class CudaDeviceMemory<T>						*
************************************************************************/
template <class T>
class CudaDeviceMemory : private Buf<T>
{
  public:
    typedef T			value_type;	  //!< 要素の型
    typedef value_type*		pointer;	  //!< 要素へのポインタ
    typedef const value_type*	const_pointer;	  //!< 定数要素へのポインタ

  public:
    explicit CudaDeviceMemory(u_int siz=0)				;
    ~CudaDeviceMemory()							;

    using		Buf<T>::operator T*;
    using		Buf<T>::operator const T*;
    using		Buf<T>::size;
    using		Buf<T>::dim;
    using		Buf<T>::align;
    
    template <class T2, class B2> CudaDeviceMemory&
			readFrom(const Array<T2, B2>& a)		;
    template <class T2, class B2> const CudaDeviceMemory&
			writeTo(Array<T2, B2>& a)		const	;
    bool		resize(u_int siz)				;
    void		resize(T* p, u_int siz)				;
    
  private:
    CudaDeviceMemory(const CudaDeviceMemory&)				;
    CudaDeviceMemory&	operator =(const CudaDeviceMemory&)		;
    
    static T*		memalloc(u_int siz)				;
    static void		memfree(T* p)					;
};

template <class T> inline
CudaDeviceMemory<T>::CudaDeviceMemory(u_int siz)
    :Buf<T>(memalloc(siz), siz)
{
}

template <class T> inline
CudaDeviceMemory<T>::~CudaDeviceMemory()
{
    memfree((T*)*this);
}
    
template <class T>
template <class T2, class B2> inline CudaDeviceMemory<T>&
CudaDeviceMemory<T>::readFrom(const Array<T2, B2>& a)
{
    using namespace	std;
    
    if (sizeof(T) != sizeof(T2))
	throw logic_error(
	    "CudaDeviceMemory<T>::readFrom: mismatched element sizes!!");
    if (dim() != a.dim())
	throw invalid_argument(
	    "CudaDeviceMemory<T>::readFrom: mismatched dimensions!!");
    CUDA_SAFE_CALL(cudaMemcpy(pointer(*this), (const T2*)a, 
			      dim() * sizeof(T), cudaMemcpyHostToDevice));
    return *this;
}
    
template <class T>
template <class T2, class B2> inline const CudaDeviceMemory<T>&
CudaDeviceMemory<T>::writeTo(Array<T2, B2>& a) const
{
    using namespace	std;
    
    if (sizeof(T) != sizeof(T2))
	throw logic_error(
	    "CudaDeviceMemory<T>::writeTo: mismatched element sizes!!");
    a.resize(dim());
    CUDA_SAFE_CALL(cudaMemcpy((T2*)a, const_pointer(*this),
			      dim() * sizeof(T), cudaMemcpyDeviceToHost));
    return *this;
}

template <class T> inline bool
CudaDeviceMemory<T>::resize(u_int siz)
{
    if (siz == size())
	return false;

    memfree((T*)*this);
    Buf<T>::resize(memalloc(siz), siz);
    return true;
}

template <class T> inline void
CudaDeviceMemory<T>::resize(T* p, u_int siz)
{
    Buf<T>::resize(p, siz);
}

template <class T> inline T*
CudaDeviceMemory<T>::memalloc(u_int siz)
{
    using namespace	std;
    
    T*	p = 0;
    if (siz > 0)
    {
	CUDA_SAFE_CALL(cudaMalloc((void**)&p, siz*sizeof(T)));
	if (p == 0)
	    throw runtime_error("Failed to allocate CUDA device memoery!!");
    }
    return p;
}

template <class T> inline void
CudaDeviceMemory<T>::memfree(T* p)
{
    if (p != 0)
	CUDA_SAFE_CALL(cudaFree(p));
}
    
/************************************************************************
*  class CudaDeviceMemory2<T>						*
************************************************************************/
template <class T, class R=Buf<CudaDeviceMemory<T> > >
class CudaDeviceMemory2 : public Array2<CudaDeviceMemory<T>,
					CudaDeviceMemory<T>, R>
{
  private:
    typedef Array2<CudaDeviceMemory<T>, CudaDeviceMemory<T>, R>	super;
    
  public:
    typedef CudaDeviceMemory<T>	row_type;	  //!< 行の型
    typedef R			rowbuffer_type;	  //!< 行バッファの型
    typedef CudaDeviceMemory<T>	buffer_type;	  //!< バッファの型
    typedef T			value_type;	  //!< 要素の型
    typedef ptrdiff_t		difference_type;  //!< ポインタ間の差
    typedef value_type&		reference;	  //!< 要素への参照
    typedef const value_type&	const_reference;  //!< 定数要素への参照
    typedef value_type*		pointer;	  //!< 要素へのポインタ
    typedef const value_type*	const_pointer;	  //!< 定数要素へのポインタ

  public:
    CudaDeviceMemory2()							;
    CudaDeviceMemory2(u_int r, u_int c)					;

    using	super::operator T*;
    using	super::operator const T*;
    using	super::begin;
    using	super::end;
    using	super::size;
    using	super::dim;
    using	super::nrow;
    using	super::ncol;
    
    template <class T2, class B2, class R2> CudaDeviceMemory2&
		readFrom(const Array2<T2, B2, R2>& a)			;
    template <class T2, class B2, class R2> const CudaDeviceMemory2&
		writeTo(Array2<T2, B2, R2>& a)			const	;
};

template <class T, class R> inline
CudaDeviceMemory2<T, R>::CudaDeviceMemory2()
    :Array2<CudaDeviceMemory<T>, CudaDeviceMemory<T>, R>()
{
}

template <class T, class R> inline
CudaDeviceMemory2<T, R>::CudaDeviceMemory2(u_int r, u_int c)
    :Array2<CudaDeviceMemory<T>, CudaDeviceMemory<T>, R>(r, c)
{
}

template <class T, class R>
template <class T2, class B2, class R2> CudaDeviceMemory2<T, R>&
CudaDeviceMemory2<T, R>::readFrom(const Array2<T2, B2, R2>& a)
{
    check_dim(a.nrow());
    for (int i = 0; i < nrow(); ++i)
	(*this)[i].readFrom(a[i]);
    return *this;
}

template <class T, class R>
template <class T2, class B2, class R2> const CudaDeviceMemory2<T, R>&
CudaDeviceMemory2<T, R>::writeTo(Array2<T2, B2, R2>& a) const
{
    a.resize(nrow(), ncol());
    for (int i = 0; i < nrow(); ++i)
	(*this)[i].writeTo(a[i]);
    return *this;
}

}

#endif	/* !__TUCudaDeviceMemory_h */
