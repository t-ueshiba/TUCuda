/*!
  \file		ICIA.h
  \author	Toshio UESHIBA
  \brief	クラス TU::cuda::ICIA の定義と実装
*/
#ifndef TU_ICIA_H
#define TU_ICIA_H

#include "TU/Geometry++.h"
#include "TU/Image++.h"
#include "TU/cuda/FIRGaussianConvolver.h"
#include "TU/cuda/chrono.h"
#include <iomanip>

namespace TU
{
namespace cuda
{
#if defined(__NVCC__)
namespace device
{
/************************************************************************
*  __global__ functions							*
************************************************************************/
template <class MAP, class EDGE, class GRAD, class MOMENT, class STRIDE_M>
__global__ void
compute_grad_and_moment(EDGE edgeH, EDGE edgeV, GRAD grad, MOMENT moment,
			int strideE, int strideG, STRIDE_M strideM)
{
    const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int	p = y*strideE + x;
    const auto	g = MAP::grad(edgeH[p], edgeV[p], x, y);

    grad[y*strideG + x] = g;
    advance_stride(moment, y*strideM);
    moment[x] = MAP::moment(g);
}

template <>
__global__ void
sqrerr(IN src, IN dst, GRAD grad, int strideI, int strideG)
{
    const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

    const auto	sval = src[y*strideI + x];
    const auto	dval = at(dst, );

    if (sval > 0.5 && dval > 0.5)
    {
	const auto	dI = clamp(dval - sval,
				   -intensityThresh, intensityThresh);
	sqr = dI * dI;
	g   = dI * grad[y*strideG + x];
    }
    else
    {
	sqr = 0;
	set_zero(g);
    }
}
    
}	//namespce device
#endif	// __NVCC__
    
/************************************************************************
*  class ICIA<MAP, CLOCK>						*
************************************************************************/
template <class MAP, class CLOCK=void>
class ICIA : public Profiler<CLOCK>
{
  public:
    using value_type	= typename MAP::element_type;

    struct Parameters
    {
	Parameters()
	    :sigma(1.0), newton(false), niter_max(100),
	     intensityThresh(15), tol(1.0e-4)				{}

	float		sigma;
	bool		newton;
	size_t		niter_max;
	value_type	intensityThresh;
	value_type	tol;
    };
    
  private:
    using super		= Profiler<CLOCK>;
    using params_type	= Array< value_type, MAP::DOF>;
    using matrix_type	= Array2<value_type, MAP::DOF, MAP::DOF>;
    
  public:
    using	super::start;
    using	super::nextFrame;

    ICIA(const Parameters& params=Parameters())
	:super(3), _params(params)					{}

    template <class IMAGE>
    void	initialize(const IMAGE& src)				;
    template <class IMAGE>
    void	initialize(const IMAGE& edgeH, const IMAGE& edgeV)	;
    template <class IMAGE>
    value_type	operator ()(const IMAGE& src, const IMAGE& dst, MAP& f,
			    size_t u0=0, size_t v0=0,
			    size_t w=0, size_t h=0)			;

  private:
    template <class IMAGE>
    value_type	sqrerr(const IMAGE& src, const IMAGE& dst,
		       const MAP& f, params_type& g,
		       size_t u0, size_t v0, size_t w, size_t h) const	;
    matrix_type	moment(size_t u0, size_t v0, size_t w, size_t h) const	;

  private:
    Parameters		_params;
    Array2<params_type>	_grad;
    Array2<matrix_type>	_M;
};

template <class MAP, class CLOCK> template <class IMAGE> void
ICIA<MAP, CLOCK>::initialize(const IMAGE& src)
{
    using	std::cbegin;
    using	std::cend;
    
    start(0);
  // 位置に関する原画像の輝度勾配を求める．
    Array2<value_type>			src_d(src);
    Array2<value_type>			edgeH(size<0>(src), size<1>(src));
    Array2<value_type>			edgeV(size<0>(src), size<1>(src));
    FIRGaussianConvolver2<value_type>	convolver(_params.sigma);
    convolver.diffH(cbegin(src_d), cend(src_d), edgeH.begin());
    convolver.diffV(cbegin(src_d), cend(src_d), edgeV.begin());

    initialize(edgeH, edgeV);
}

template <class MAP, class CLOCK> template <class IMAGE> void
ICIA<MAP, CLOCK>::initialize(const IMAGE& edgeH, const IMAGE& edgeV)
{
    _grad.resize(size<0>(edgeH), size<1>(edgeH));
    _M   .resize(size<0>(edgeH), size<1>(edgeH));

    start(1);
  // 変換パラメータに関する原画像の輝度勾配とモーメント行列を求める．
    for (size_t v = 0; v < _grad.nrow(); ++v)
    {
	using	std::cbegin;
	using	std::begin;
	
	auto	eH   = cbegin(edgeH[v]);
	auto	eV   = cbegin(edgeV[v]);
	auto	grad = begin(_grad[v]);
	auto	M    = begin(_M[v]);
	
	if (v == 0)
	{
	    matrix_type	val(MAP::DOF, MAP::DOF);
	    
	    for (size_t u = 0; u < _grad.ncol(); ++u)
	    {
		const auto	J = MAP::derivative0(u, v);
		*grad = *eH * J[0] + *eV * J[1];
		val  += *grad % *grad;
		*M    = val;
		++eH;
		++eV;
		++grad;
		++M;
	    }
	}
	else
	{
	    matrix_type	val(MAP::DOF, MAP::DOF);
	    auto	Mp = _M[v-1].cbegin();

	    for (size_t u = 0; u < _grad.ncol(); ++u)
	    {
		const auto	J = MAP::derivative0(u, v);
		*grad = *eH * J[0] + *eV * J[1];
		val  += *grad % *grad;
		*M    = val + *Mp;
		++eH;
		++eV;
		++grad;
		++M;
		++Mp;
	    }
	}
    }
}
    
template <class MAP, class CLOCK> template <class IMAGE> auto
ICIA<MAP, CLOCK>::operator ()(const IMAGE& src, const IMAGE& dst, MAP& f,
			      size_t u0, size_t v0, size_t w, size_t h)
    -> value_type
{
#ifdef ICIA_DEBUG
    std::cout << 'M' << 2 << std::endl;
    src.saveHeader(std::cout, ImageFormat::RGB_24);
    src.saveHeader(std::cout, ImageFormat::U_CHAR);
#endif
    start(2);

    if (w == 0)
	w = size<1>(src) - u0;
    if (h == 0)
	h = size<0>(src) - v0;
    
    if (_params.newton)
    {
	const auto	Minv = inverse(moment(u0, v0, w, h));
	value_type	sqr = 0;
	for (size_t n = 0; n < _params.niter_max; ++n)
	{
	    params_type	g;
	    const auto	sqr_new = sqrerr(src, dst, f, g, u0, v0, w, h);
#ifdef _DEBUG
	    std::cerr << "[" << std::setw(2) << n << "] sqr = " << sqr_new
		      << std::endl;
#endif
	    f.compose(Minv*g);
	    if (fabs(sqr - sqr_new) <= _params.tol*(sqr_new + sqr + 1.0e-7))
	    {
		nextFrame();
		return sqr_new;
	    }
	    sqr = sqr_new;
	}
    }
    else
    {
	auto		M = moment(u0, v0, w, h);
	params_type	diagM(M.size());
	for (size_t i = 0; i < diagM.size(); ++i)
	    diagM[i] = M[i][i];
	
	params_type	g;
	auto		sqr = sqrerr(src, dst, f, g, u0, v0, w, h);
#ifdef _DEBUG
	std::cerr << "     sqr = " << sqr << std::endl;
#endif
	value_type	lambda = 1.0e-4;
	for (size_t n = 0; n < _params.niter_max; ++n)	// L-M反復
	{
	    for (size_t i = 0; i < M.size(); ++i)
		M[i][i] = (1.0 + lambda) * diagM[i];

	    auto	dtheta(g);
	    solve(M, dtheta);
	    auto	f_new(f);
	    f_new.compose(dtheta);
	    params_type	g_new;
	    const auto	sqr_new = sqrerr(src, dst, f_new, g_new, u0, v0, w, h);
#ifdef _DEBUG	    
	    std::cerr << "[" << std::setw(2) << n << "] sqr = " << sqr_new
		 << ", sqr_old = " << sqr
		 << ",\tlambda = " << lambda << std::endl;
#endif
	    if (sqr_new <= sqr)		// 二乗誤差が減少するか調べる．
	    {
		f   = f_new;

	      // 収束判定
		if (fabs(sqr - sqr_new) <= _params.tol*(sqr_new + sqr + 1.0e-7))
		{
		    nextFrame();
		    return sqr_new;
		}
		
		g   = g_new;
		sqr = sqr_new;
		lambda *= 0.1;		// L-M反復のパラメータを減らす．
	    }
	    else if (lambda < 1.0e-10)
	    {
		nextFrame();
		return sqr;
	    }
	    else
		lambda *= 10.0;		// L-M反復のパラメータを増やす．
	}
    }

    throw std::runtime_error("ICIA::operator (): maximum iteration limit exceeded!");

    return -1.0;
}
    
template <class MAP, class CLOCK> template <class IMAGE> auto
ICIA<MAP, CLOCK>::sqrerr(const IMAGE& src, const IMAGE& dst,
			 const MAP& f, params_type& g,
			 size_t u0, size_t v0, size_t w, size_t h) const
    -> value_type
{
#ifdef ICIA_DEBUG
    Image<RGB>		rgbImage(size<1>(src), size<0>(src));
    Image<u_char>	composedImage(size<1>(src), size<0>(src));
#endif
    g.resize(MAP::DOF);
    g = 0;
    
    value_type	sqr = 0.0;
    size_t	npoints = 0;
    for (size_t v = v0; v < v0 + h; ++v)
    {
	using	std::cbegin;
	
	auto	sval = cbegin(src[v]) + u0;
	auto	grad = _grad[v].cbegin() + u0;
		
	for (size_t u = u0; u < u0 + w; ++u)
	{
	    const auto	p = f(u, v);

	    if (0 <= p[0] && p[0] < dst.ncol() - 1 &&
		0 <= p[1] && p[1] < dst.nrow() - 1)
	    {
		const auto	dval = at(dst, p[0], p[1]);
		if (dval > 0.5 && *sval > 0.5)
		{
		    auto	dI = dval - *sval;
		    if (dI > _params.intensityThresh)
			dI = _params.intensityThresh;
		    else if (dI < -_params.intensityThresh)
			dI = -_params.intensityThresh;
#ifdef ICIA_DEBUG
		    if (dI > 0.0)
			rgbImage[v][u]
			    = RGB(0, 255*dI/_params.intensityThresh, 0);
		    else
			rgbImage[v][u]
			    = RGB(-255*dI/_params.intensityThresh, 0, 0);
		    composedImage[v][u] = (dval + *sval) / 2;
#endif
		    g   += dI * *grad;
		    sqr += dI * dI;
		    ++npoints;
		}
#ifdef ICIA_DEBUG
		else
		    rgbImage[v][u] = RGB(0, 0, 255);
#endif
	    }
	    ++sval;
	    ++grad;
	}
    }
#ifdef ICIA_DEBUG
    rgbImage.saveData(std::cout, ImageFormat::RGB_24);
    composedImage.saveData(std::cout, ImageFormat::U_CHAR);
#endif
    if (npoints < MAP::DOF)
	throw std::runtime_error("ICIA::sqrerr(): not enough points!");
    
    return sqr / npoints;
}
    
template <class MAP, class CLOCK> auto
ICIA<MAP, CLOCK>::moment(size_t u0, size_t v0, size_t w, size_t h) const
    -> matrix_type
{
    auto	u1 = std::min(u0 + w, _M.ncol()) - 1;
    auto	v1 = std::min(v0 + h, _M.nrow()) - 1;
    matrix_type	val;

    if (u0 < _M.ncol() && v0 < _M.nrow() && u1 > 0 && v1 > 0)
    {
	if (u0-- > 0)
	{
	    if (v0-- > 0)
		val = _M[v1][u1] - _M[v1][u0] + _M[v0][u0] - _M[v0][u1];
	    else
		val = _M[v1][u1] - _M[v1][u0];
	}
	else
	{
	    if (v0-- > 0)
		val = _M[v1][u1] - _M[v0][u1];
	    else
		val = _M[v1][u1];
	}
    }

    return val;
}
    
}	// namespace cuda
}	// namespace TU
#endif	// !TU_ICIA_H
