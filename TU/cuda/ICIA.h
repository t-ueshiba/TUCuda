/*!
  \file		ICIA.h
  \author	Toshio UESHIBA
  \brief	クラス TU::cuda::ICIA の定義と実装
*/
#pragma once

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
namespace detail
{
  template <class OUT, class VEC> __device__
  std::enable_if_t<ncol<VEC>() == 1, OUT>
  assign_grad(OUT out, const VEC& v, int stride)
  {
      *out = v;
      advance_stride(out, stride);
      return out;
  }

  template <class OUT, class T, size_t C> __device__ OUT
  assign_grad(OUT out, const Mat2x<T, C>& m, int stride)
  {
      return assign_grad(assign_grad(out, m.x, stride), m.y, stride);
  }

  template <class OUT, class T, size_t C> __device__ OUT
  assign_grad(OUT out, const Mat3x<T, C>& m, int stride)
  {
      return assign_grad(assign_grad(assign_grad(out, m.x, stride),
				     m.y, stride),
			 m.z, stride);
  }

  template <class OUT, class T, size_t C> __device__ OUT
  assign_grad(OUT out, const Mat4x<T, C>& m, int stride)
  {
      return assign_grad(assign_grad(assign_grad(assign_vec(out, m.x, stride),
						 m.y, stride),
				     m.z, stride),
			 m.w, stride);
  }

  template <class OUT, class VEC> __device__ OUT
  std::enable_if_t<ncol<VEC>() == 1, OUT>
  assign_moment(OUT out, const VEC& a, const VEC& b, int stride)
  {
      return assign_grad(out, ext(a, b), stride);
  }

  template <class OUT, class T, size_t C> __device__ void
  assign_moment(OUT out, const Mat2x<T, C>& a, const Mat2x<T, C>& b, int stride)
  {
      assign_moment(assign_moment(out, a.x, b.x, 2*stride),
		    a.y, b.x, 2*stride);
      assign_moment(assign_moment(out + stride, a.x, b.y, 2*stride),
		    a.y, b.y, 2*stride);
  }

  __device__ float
  clamp(float x, float lower, float upper)
  {
      return fminf(fmaxf(x, lower), upper);
  }

  __device__ double
  clamp(double x, double lower, double upper)
  {
      return fmin(fmax(x, lower), upper);
  }
}	// namespsace detail

/************************************************************************
*  __global__ functions							*
************************************************************************/
template <class MAP, class EDGE, class GRAD, class MOMENT>
__global__ void
compute_grad_and_moment(EDGE edgeH, EDGE edgeV, GRAD grad, MOMENT moment,
			int strideE, int strideG, int strideM)
{
    const auto	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const auto	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const auto	p = __mul24(y, strideE) + x;
    const auto	g = MAP::image_derivative0(edgeH[p], edgeV[p], x, y);

    advance_stride(grad, y*strideG);
    detail::assign_grad(grad + x, g, strideG);
    advance_stride(moment, y*strideM);
    detail::assign_moment(moment + x, g, g, strideM);
}

template <class IN, class GRAD, class MAP>
__global__ void
sqrerr(IN src, cudaTextureObject_t dst, GRAD grad, MAP map,
       int x0, int y0, int strideI, int strideG)
{
    using value_t = iterator_value<IN>;

    const int	x = x0 + __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	y = y0 + __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const auto	p = map(x, y);

    advance_stride(src, y*strideI);

    const auto	sval = src[x];
    const auto	dval = tex2D<value_t>(dst, p.x, p.y);

    if (sval > 0.5 && dval > 0.5)
    {
	const auto	dI = detail::clamp(dval - sval,
					   -intensityThresh, intensityThresh);
	sqr = dI * dI;
	dIg = dI * grad[y*strideG + x];
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
  private:
    template <class MAP_>	struct map_traits;
    template <>
    struct map_traits<Projectivity<float, 2, 2> >
    {
	using element_type	= float;
	using vec_type		= float4;

	constexpr static size_t	GRAD_NVECS   = 2;
	constexpr static size_t	MOMENT_NVECS = 16;
    };
    template <class T>
    struct map_traits<Affinity<T, 2, 2> >
    {
	using element_type	= float;
	using vec_type		= float3;

	constexpr static size_t	GRAD_NVECS   = 2;
	constexpr static size_t	MOMENT_NVECS = 6;
    };
    template <class T>
    struct map_traits<Rigidity<T, 2> >
    {
	using element_type	= float;
	using vec_type		= float3;

	constexpr static size_t	GRAD_NVECS   = 1;
	constexpr static size_t	MOMENT_NVECS = 3;
    };

    constexpr static size_t	GRAD_NVECS   = map_traits<MAP>::GRAD_NVECS;
    constexpr static size_t	MOMENT_NVECS = map_traits<MAP>::MOMENT_NVECS;


  public:
    using element_type	= typename MAP::element_type;
    using value_type	= typename MAP::value_type;

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

    template <class T_>
    void	initialize(const Array2<T_>& src)			;
    void	initialize(const Array2<value_type>& edgeH,
			   const Array2<value_type>& edgeV)		;
    template <class T_>
    value_type	operator ()(const Array2<T_>& src,
			    const Array2<T_>& dst, MAP& f,
			    size_t u0=0, size_t v0=0,
			    size_t w=0, size_t h=0)			;

  private:
    template <class T_>
    value_type	sqrerr(const Array2<T_>& src, const Array2<T_>& dst,
		       const MAP& f, params_type& g,
		       size_t u0, size_t v0, size_t w, size_t h) const	;
    matrix_type	moment(size_t u0, size_t v0, size_t w, size_t h) const	;

  private:
    Parameters			_params;
    Array3<param_vec_type>	_grad;
    Array3<param_vec_type>	_M;
};

template <class MAP, class CLOCK> template <class Array2<T_>> void
ICIA<MAP, CLOCK>::initialize(const Array2<T_>& src)
{
    using	std::cbegin;
    using	std::cend;

    start(0);
  // 位置に関する原画像の輝度勾配を求める．
    Array2<value_type>			edgeH(size<0>(src), size<1>(src));
    Array2<value_type>			edgeV(size<0>(src), size<1>(src));
    FIRGaussianConvolver2<value_type>	convolver(_params.sigma);
    convolver.diffH(cbegin(src), cend(src), edgeH.begin());
    convolver.diffV(cbegin(src), cend(src), edgeV.begin());

    initialize(edgeH, edgeV);
}

template <class MAP, class CLOCK> void
ICIA<MAP, CLOCK>::initialize(const Array2<value_type>& edgeH,
			     const Array2<value_type>& edgeV)
{
    const auto	nrow = size<0>(edgeH);
    const auto	ncol = size<1>(edgeH);
    _grad.resize(GRAD_NVECS,   nrow, ncol);
    _M   .resize(MOMENT_NVECS, nrow, ncol);

    const auto	strideH = edgeH.stride();
    const auto	strideV = edgeV.stride();

    const auto	stride_o = stride(out);

    start(1);
  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::warp<T><<<blocks, threads>>>(tex.get(), begin(*out),
					 op, 0, 0, stride_o);
  // 右上
    const auto	x0 = blocks.x*threads.x;
    threads.x = ncol - x0;
    blocks.x  = 1;
    device::warp<T><<<blocks, threads>>>(tex.get(), begin(*out),
					 op, x0, 0, stride_o);
  // 左下
    const auto	y0 = blocks.y*threads.y;
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow - y0;
    blocks.y  = 1;
    device::warp<T><<<blocks, threads>>>(tex.get(), begin(*out),
					 op, 0, y0, stride_o);
  // 右下
    threads.x = ncol - x0;
    blocks.x  = 1;
    device::warp<T><<<blocks, threads>>>(tex.get(), begin(*out),
					 op, x0, y0, stride_o);
}

template <class MAP, class CLOCK> template <class T_> auto
ICIA<MAP, CLOCK>::operator ()(const Array2<T_>& src, const Array2<T_>& dst,
			      MAP& f, size_t u0, size_t v0, size_t w, size_t h)
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

template <class MAP, class CLOCK> template <class T_> auto
ICIA<MAP, CLOCK>::sqrerr(const Array2<T_>& src, const Array2<T_>& dst,
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
