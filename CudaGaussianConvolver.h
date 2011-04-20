/*
 *  $Id: CudaGaussianConvolver.h,v 1.4 2011-04-20 08:15:07 ueshiba Exp $
 */
#include "TU/CudaFilter.h"

namespace TU
{
/************************************************************************
*  class CudaGaussianConvolver2						*
************************************************************************/
//! CUDA���Ѥ���Gauss�ˤˤ��2����������߹��ߤ�Ԥ����饹
class CudaGaussianConvolver2 : public CudaFilter2
{
  public:
    CudaGaussianConvolver2(float sigma=1.0)				;

    CudaGaussianConvolver2&	initialize(float sigma)			;

    template <class S, class T> CudaGaussianConvolver2&
	smooth(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffH(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffV(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffHH(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffHV(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffVV(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    
  private:
    Array<float>	_lobe0;		//!< ���ࡼ���󥰤Τ���Υ���
    Array<float>	_lobe1;		//!< 1����ʬ�Τ���Υ���
    Array<float>	_lobe2;		//!< 2����ʬ�Τ���Υ���
};
    
//! Gauss�ˤ���������
/*!
  \param sigma	Gauss�ˤΥ�������
*/
inline
CudaGaussianConvolver2::CudaGaussianConvolver2(float sigma)
{
    initialize(sigma);
}

//! Gauss�ˤˤ�륹�ࡼ����
/*!
  \param in	����2��������
  \param out	����2��������
  \return	����Gauss�˼���
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::smooth(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe0, _lobe0).convolve(in, out);

    return *this;
}
    
//! Gauss�ˤˤ�벣����1����ʬ(DOG)
/*!
  \param in	����2��������
  \param out	����2��������
  \return	����Gauss�˼���
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffH(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe1, _lobe0).convolve(in, out);

    return *this;
}
    
//! Gauss�ˤˤ�������1����ʬ(DOG)
/*!
  \param in	����2��������
  \param out	����2��������
  \return	����Gauss�˼���
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffV(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe0, _lobe1).convolve(in, out);

    return *this;
}
    
//! Gauss�ˤˤ�벣����2����ʬ
/*!
  \param in	����2��������
  \param out	����2��������
  \return	����Gauss�˼���
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffHH(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe2, _lobe0).convolve(in, out);

    return *this;
}
    
//! Gauss�ˤˤ��Ĳ�ξ����2����ʬ
/*!
  \param in	����2��������
  \param out	����2��������
  \return	����Gauss�˼���
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffHV(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe1, _lobe1).convolve(in, out);

    return *this;
}
    
//! Gauss�ˤˤ�������2����ʬ
/*!
  \param in	����2��������
  \param out	����2��������
  \return	����Gauss�˼���
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffVV(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe0, _lobe2).convolve(in, out);

    return *this;
}
    
}
