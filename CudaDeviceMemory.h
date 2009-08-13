/*
 *  $BJ?@.(B14-21$BG/!JFH!K;:6H5;=QAm9g8&5f=j(B $BCx:n8"=jM-(B
 *  
 *  $BAO:n<T!'?"<G=SIW(B
 *
 *  $BK\%W%m%0%i%`$O!JFH!K;:6H5;=QAm9g8&5f=j$N?&0w$G$"$k?"<G=SIW$,AO:n$7!$(B
 *  $B!JFH!K;:6H5;=QAm9g8&5f=j$,Cx:n8"$r=jM-$9$kHkL)>pJs$G$9!%Cx:n8"=jM-(B
 *  $B<T$K$h$k5v2D$J$7$KK\%W%m%0%i%`$r;HMQ!$J#@=!$2~JQ!$Bh;0<T$X3+<($9$k(B
 *  $BEy$N9T0Y$r6X;_$7$^$9!%(B
 *  
 *  $B$3$N%W%m%0%i%`$K$h$C$F@8$8$k$$$+$J$kB;32$KBP$7$F$b!$Cx:n8"=jM-<T$*(B
 *  $B$h$SAO:n<T$O@UG$$rIi$$$^$;$s!#(B
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
//! CUDA$B$K$*$$$F%G%P%$%9B&$K3NJ]$5$l$k2DJQD9%P%C%U%!%/%i%9(B
/*!
  $BC1FH$G;HMQ$9$k$3$H$O$J$/!$(B#TU::CudaDeviceMemory$B$^$?$O(B#TU::Array2$B$N(B
  $BBh(B2$B%F%s%W%l!<%H0z?t$K;XDj$9$k$3$H$K$h$C$F!$$=$l$i$N4pDl%/%i%9$H$7$F;H$&!%(B
  $B30It$K3NJ]$7$?5-21NN0h$r3d$jEv$F$k$3$H$O$G$-$J$$!%(B
  \param T	$BMWAG$N7?(B
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

    enum		{ALIGN = 32};	//!< thread$B?t(B/warp
};

//! $B;XDj$7$?MWAG?t$N%P%C%U%!$r@8@.$9$k!%(B
/*!
  \param siz	$BMWAG?t(B
*/
template <class T> inline
CudaBuf<T>::CudaBuf(u_int siz)
    :Buf<T>(memalloc(siz), siz)
{
}

//! $B%3%T!<%3%s%9%H%i%/%?(B
template <class T> inline
CudaBuf<T>::CudaBuf(const CudaBuf& b)
    :Buf<T>(memalloc(b.size()), b.size())
{
    CUDA_SAFE_CALL(cudaMemcpy((T*)*this, (const T*)b,
			      size()*sizeof(T), cudaMemcpyDeviceToDevice));
}
    
//! $BI8=`BeF~1i;;;R(B
template <class T> inline CudaBuf<T>&
CudaBuf<T>::operator =(const CudaBuf& b)
{
    resize(b.size());
    CUDA_SAFE_CALL(cudaMemcpy((T*)*this, (const T*)b,
			      size()*sizeof(T), cudaMemcpyDeviceToDevice));
    return *this;
}
    
//! $B%G%9%H%i%/%?(B
template <class T> inline
CudaBuf<T>::~CudaBuf()
{
    memfree((T*)*this);
}
    
//! $B%P%C%U%!$NMWAG?t$rJQ99$9$k!%(B
/*!
  \param siz			$B?7$7$$MWAG?t(B
  \return			siz$B$,85$NMWAG?t$HEy$7$1$l$P(Btrue$B!$$=$&$G$J$1$l$P(B
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

//! $B;XDj$5$l$?MWAG?t$r;}$D5-21NN0h$r3NJ]$9$k$?$a$K<B:]$KI,MW$JMWAG?t$rJV$9!%(B
/*!
  $B!J5-21MFNL$G$O$J$/!KMWAG?t$,(B32$B$NG\?t$K$J$k$h$&!$M?$($i$l$?MWAG?t$r7+$j>e$2$k!%(B
  \param siz	$BMWAG?t(B
  \return	32$B$NG\?t$K7+$j>e$2$i$l$?MWAG?t(B
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
//! CUDA$B$K$*$$$F%G%P%$%9B&$K3NJ]$5$l$k(BT$B7?%*%V%8%'%/%H$N(B1$B<!85%a%b%jNN0h%/%i%9(B
/*!
  \param T	$BMWAG$N7?(B
  \param B	$B%P%C%U%!(B
*/
template <class T, class B=CudaBuf<T> >
class CudaDeviceMemory : private Array<T, B>
{
  public:
    typedef T			value_type;	  //!< $BMWAG$N7?(B
    typedef ptrdiff_t		difference_type;  //!< $B%]%$%s%?4V$N:9(B
    typedef value_type*		pointer;	  //!< $BMWAG$X$N%]%$%s%?(B
    typedef const value_type*	const_pointer;	  //!< $BDj?tMWAG$X$N%]%$%s%?(B
    
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

//! CUDA$B%G%P%$%9%a%b%jNN0h$r@8@.$9$k!%(B
template <class T, class B>
CudaDeviceMemory<T, B>::CudaDeviceMemory()
    :Array<T, B>()
{
}

//! $B;XDj$7$?MWAG?t$N(BCUDA$B%G%P%$%9%a%b%jNN0h$r@8@.$9$k!%(B
/*!
  \param d	$B%a%b%jNN0h$NMWAG?t(B
*/
template <class T, class B>
CudaDeviceMemory<T, B>::CudaDeviceMemory(u_int d)
    :Array<T, B>(d)
{
}

//! $B%[%9%HB&$NG[Ns$r$3$N%G%P%$%9B&%a%b%jNN0h$KFI$_9~$`!%(B
/*!
  \param a	$B%3%T!<85$NG[Ns(B
  \return	$B$3$N%a%b%jNN0h(B
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
    
//! $B$3$N%G%P%$%9B&%a%b%jNN0h$NFbMF$rNN0h%[%9%HB&$NG[Ns$K=q$-=P$9!%(B
/*!
  \param a	$B%3%T!<@h$NG[Ns(B
  \return	$B$3$N%a%b%jNN0h(B
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
//! CUDA$B$K$*$$$F%G%P%$%9B&$K3NJ]$5$l$k(BT$B7?%*%V%8%'%/%H$N(B2$B<!85%a%b%jNN0h%/%i%9(B
/*!
  \param T	$BMWAG$N7?(B
  \param R	$B9T%P%C%U%!(B
*/
template <class T, class R=Buf<CudaDeviceMemory<T, Buf<T> > > >
class CudaDeviceMemory2
    : private Array2<CudaDeviceMemory<T, Buf<T> >, CudaBuf<T>, R>
{
  private:
    typedef Array2<CudaDeviceMemory<T, Buf<T> >, CudaBuf<T>, R>	super;
    
  public:
    typedef CudaDeviceMemory<T,Buf<T> >	row_type;	//!< $B9T$N7?(B
    typedef R				rowbuffer_type;	//!< $B9T%P%C%U%!$N7?(B
    typedef CudaBuf<T>			buffer_type;	//!< $B%P%C%U%!$N7?(B
    typedef T				value_type;	//!< $BMWAG$N7?(B
    typedef ptrdiff_t			difference_type;//!< $B%]%$%s%?4V$N:9(B
    typedef value_type*			pointer;	//!< $BMWAG$X$N%]%$%s%?(B
    typedef const value_type*		const_pointer;	//!< $BDj?tMWAG$X$N%]%$%s%?(B

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

//! 2$B<!85(BCUDA$B%G%P%$%9%a%b%jNN0h$r@8@.$9$k!%(B
template <class T, class R> inline
CudaDeviceMemory2<T, R>::CudaDeviceMemory2()
    :super()
{
}

//! $B9T?t$HNs?t$r;XDj$7$F(B2$B<!85(BCUDA$B%G%P%$%9%a%b%jNN0h$r@8@.$9$k!%(B
/*!
  \param r	$B9T?t(B
  \param c	$BNs?t(B
*/
template <class T, class R> inline
CudaDeviceMemory2<T, R>::CudaDeviceMemory2(u_int r, u_int c)
    :super(r, c)
{
}

//! $B%3%T!<%3%s%9%H%i%/%?(B
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
    
//! $BI8=`BeF~1i;;;R(B
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
    
//! $B%[%9%HB&$NG[Ns$r$3$N%G%P%$%9B&%a%b%jNN0h$KFI$_9~$`!%(B
/*!
  \param a	$B%3%T!<85$NG[Ns(B
  \return	$B$3$N%a%b%jNN0h(B
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

//! $B$3$N%G%P%$%9B&%a%b%jNN0h$NFbMF$rNN0h%[%9%HB&$NG[Ns$K=q$-=P$9!%(B
/*!
  \param a	$B%3%T!<@h$NG[Ns(B
  \return	$B$3$N%a%b%jNN0h(B
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
