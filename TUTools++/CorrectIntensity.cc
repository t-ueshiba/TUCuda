/*
 *  $BJ?@.(B14-19$BG/!JFH!K;:6H5;=QAm9g8&5f=j(B $BCx:n8"=jM-(B
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
 *  Copyright 2002-2007.
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
 *  $Id: CorrectIntensity.cc,v 1.3 2008-09-10 05:10:34 ueshiba Exp $
 */
#include "TU/CorrectIntensity.h"
#include "TU/mmInstructions.h"

namespace TU
{
#if defined(SSE)
template <class T> static inline void
mmCorrect(T* p, mmFlt a, mmFlt b)
{
    const mmInt	val = mmLoadU((mmInt*)p);
#  if defined(SSE2)
    mmStoreU((mmInt*)p,
	     mmPackUS(mmPack32(mmToInt32F(mmAddF(mmMulF(mmToFlt0(val),
							a), b)),
			       mmToInt32F(mmAddF(mmMulF(mmToFlt1(val),
							a), b))),
		      mmPack32(mmToInt32F(mmAddF(mmMulF(mmToFlt2(val),
							a), b)),
			       mmToInt32F(mmAddF(mmMulF(mmToFlt3(val),
							a), b)))));
#  else
    mmStoreU((mmInt*)p,
	     mmPackUS(mmToIntF(mmAddF(mmMulF(mmToFlt0(val), a), b)),
		      mmToIntF(mmAddF(mmMulF(mmToFlt1(val), a), b))));
#  endif
}

template <> inline void
mmCorrect(short* p, mmFlt a, mmFlt b)
{
#  if defined(SSE2)
    const mmInt	val = mmLoadU((mmInt*)p);
    mmStoreU((mmInt*)p,
	     mmPack32(mmToInt32F(mmAddF(mmMulF(mmToFltL(val), a), b)),
		      mmToInt32F(mmAddF(mmMulF(mmToFltH(val), a), b))));
#  else
    mmStoreU((mmInt*)p,
	     mmToIntF(mmAddF(mmMulF(mmToFlt(mmLoadU((mmInt*)p)), a), b)));
#  endif
}

template <> inline void
mmCorrect(float* p, mmFlt a, mmFlt b)
{
    mmStoreFU(p, mmAddF(mmMulF(mmLoadF(p), a), b));
}
#endif
/************************************************************************
*  class CorrectIntensity						*
************************************************************************/
template <class T> void
CorrectIntensity::operator()(Image<T>& image, int vs, int ve) const
{
    if (ve == 0)
	ve = image.height();
    
    for (int v = vs; v < ve; ++v)
    {
	T*		p = image[v];
	T* const	q = p + image.width();
#if defined(SSE)
	const mmFlt	a = mmSetF(_gain), b = mmSetF(_offset);
	for (T* const r = q - mmNBytes/sizeof(T);
	     p <= r; p += mmNBytes/sizeof(T))
	    mmCorrect(p, a, b);
	_mm_empty();
#endif
	for (; p < q; ++p)
	    *p = val(*p);
    }
}
    
template <class T> inline T
CorrectIntensity::val(T pixel) const
{
    float	r = _gain * pixel.r + _offset;
    r = (r < 0.0 ? 0 : r > 255.0 ? 255 : u_char(r));
    float	g = _gain * pixel.g + _offset;
    g = (g < 0.0 ? 0 : g > 255.0 ? 255 : u_char(g));
    float	b = _gain * pixel.b + _offset;
    b = (b < 0.0 ? 0 : b > 255.0 ? 255 : u_char(b));
    return T(r, g, b);
}

template <> inline u_char
CorrectIntensity::val(u_char pixel) const
{
    const float	val = _gain * pixel + _offset;
    return (val < 0.0 ? 0 : val > 255.0 ? 255 : u_char(val));
}
    
template <> inline short
CorrectIntensity::val(short pixel) const
{
    const float	val = _gain * pixel + _offset;
    return (val < 0.0 ? 0 : val > 65535.0 ? 65535 : short(val));
}
    
template <> inline float
CorrectIntensity::val(float pixel) const
{
    return pixel;
}

template void
CorrectIntensity::operator ()(Image<u_char>& image, int vs, int ve) const;
template void
CorrectIntensity::operator ()(Image<short>& image,  int vs, int ve) const;
template void
CorrectIntensity::operator ()(Image<float>& image,  int vs, int ve) const;
template void
CorrectIntensity::operator ()(Image<RGBA>& image,   int vs, int ve) const;
template void
CorrectIntensity::operator ()(Image<ABGR>& image,   int vs, int ve) const;
    
}
