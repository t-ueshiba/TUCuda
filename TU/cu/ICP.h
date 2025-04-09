// Software License Agreement (BSD License)
//
// Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of National Institute of Advanced Industrial
//    Science and Technology (AIST) nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Toshio Ueshiba
//
/*!
  \file		ICP.h
  \brief	隣接フレーム間でpoint cloudを位置合わせしてフレーム間運動を推定
*/
#pragma once

#include <Eigen/Eigen>
#include <array>
#include <TU/Profiler.h>
#include "TU/cu/array.h"
#include "TU/cu/vec.h"
#include "TU/cu/chrono.h"
#include "TU/cu/Array++.h"
#include "TU/cu/Texture.h"
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace TU::cu
{
namespace icp
{
/************************************************************************
*  class ErrorMetric<ICP>						*
************************************************************************/
template <class ICP>
class ErrorMetric
{
  public:
    using value_type		= typename ICP::value_type;
    using color_type		= typename ICP::color_type;
    using transform_type	= typename ICP::transform_type;
    using intrinsics_type	= typename ICP::intrinsics_type;
    using frame_type		= typename ICP::Frame;

    constexpr static size_t	DOF = transform_type::DOF;

    using array_type		= array<value_type, (DOF+1)*(DOF+2)/2 + 1>;
    using matrix_type		= Eigen::Matrix<value_type, DOF, DOF>;
    using vector_type		= Eigen::Matrix<value_type, DOF, 1>;

  private:
    using point_type		= typename transform_type::point_type;
    using direction_type	= typename transform_type::direction_type;
    using point2_type		= typename intrinsics_type::point2_type;
    using points_type		= range<range_iterator<
					    thrust::device_ptr<
						const point_type> > >;
    using directions_type	= range<range_iterator<
					    thrust::device_ptr<
						const direction_type> > >;
    using image_type		= range<range_iterator<
					    thrust::device_ptr<
						const color_type> > >;

  public:
    ErrorMetric(const transform_type& Tts,
		const frame_type& source, const frame_type& target,
		value_type dist_thresh, value_type angle_thresh,
		value_type color_thresh, value_type color_weight)
	:_Tts(Tts), _intrinsics(target.intrinsics),
	 _xs(source.points.cbegin(),  source.points.nrow()),
	 _ns(source.normals.cbegin(), source.normals.nrow()),
	 _image_s(source.image.cbegin(), source.image.nrow()),
	 _xt(target.points.cbegin(),  target.points.nrow()),
	 _nt(target.normals.cbegin(), target.normals.nrow()),
	 _image_t(target.image),
	 _edgeH(target.edgeH),
	 _edgeV(target.edgeV),
	 _sqdist_thresh(dist_thresh*dist_thresh),
	 _sqangle_thresh(angle_thresh*angle_thresh),
	 _sqcolor_thresh(color_thresh*color_thresh),
	 _color_weight(color_weight)
    {
    }

    __device__ __forceinline__ array_type
    operator()(int i) const
    {
	const int	v    = i / ncol();
	const int	u    = i - (v * ncol());
	const auto	xs   = _xs[v][u];
	const auto	xs_t = _Tts(xs);
	const auto	uv_t = _intrinsics(xs_t);
	const int	ut   = device::to_int(uv_t.x);
	const int	vt   = device::to_int(uv_t.y);

	if (xs.z > 0 && xs_t.z > 0 &&
	    0 <= ut && ut < ncol() && 0 <= vt && vt < nrow())
	{
	    const auto	xt   = _xt[vt][ut];
	    const auto	nt   = _nt[vt][ut];
	    const auto	ns   = _ns[v][u];
	    const auto	ns_t = _Tts.direction(ns);

	    if (nt.z > 0 && ns.z > 0 &&
		square(cross(ns_t, nt)) < _sqangle_thresh &&
		square(xt - xs_t)	< _sqdist_thresh)
	    {
		auto	m = point_plane_moment(xs_t, xt, nt)
			  + _color_weight * color_moment(
						xs_t, uv_t,
						_image_s[v][u] -
						_image_t(uv_t.x, uv_t.y));
		m[array_type::size()-1] = 1;

		return m;
	    }
	}

	return {0};
    }

    int
    size() const
    {
	return nrow() * ncol();
    }

    static matrix_type
    M(const array_type& moment)
    {
	matrix_type	m;
	auto		p = moment.data();
	for (int i = 0; i < m.rows(); ++i)
	{
	    for (int j = i; j < m.cols(); ++j)
		m(j, i) = m(i, j) = *p++;
	    ++p;				// skip deviation term
	}

	return m;
    }

    static vector_type
    d(const array_type& moment)
    {
	vector_type	v;
	v << moment[6],  moment[12], moment[17],
	     moment[21], moment[24], moment[26];

	return v;
    }

    static value_type
    npoints(const array_type& moment)
    {
	return moment[array_type::size()-1];
    }

    static value_type
    mse(const array_type& moment)
    {
	return moment[array_type::size()-2] / moment[array_type::size()-1];
    }

  private:
    __host__ __device__ __forceinline__
    int		nrow()		const	{ return _xs.size(); }
    __host__ __device__ __forceinline__
    int		ncol()		const	{ return _xs.cbegin().size(); }
	
    __device__ __forceinline__ static array_type
    point_plane_moment(const point_type& x,
		       const point_type& x_observed, const direction_type& n)
    {
	array<value_type, DOF+1>	row;
	row[0] = n.x;
	row[1] = n.y;
	row[2] = n.z;
	const auto	x_cross_n = cross(x, n);
	row[3] = x_cross_n.x;
	row[4] = x_cross_n.y;
	row[5] = x_cross_n.z;
	row[6] = dot(n, x_observed - x);	// deviation term

	return row.template ext<array_type::size()>();
    }
    
    template <class C_> __device__ __forceinline__ array_type
    color_moment(const point_type& x,
		 const point2_type& uv, const C_& color_diff) const
    {
	if (color_diff*color_diff > _sqcolor_thresh)
	    return {0};
	
	const auto a = _intrinsics.image_derivative0(x, _edgeH(uv.x, uv.y),
							_edgeV(uv.x, uv.y));
	array<value_type, DOF+1>	row;
	row[0] = a.x;
	row[1] = a.y;
	row[2] = a.z;
	const auto	x_cross_a = cross(x, a);
	row[3] = x_cross_a.x;
	row[4] = x_cross_a.y;
	row[5] = x_cross_a.z;
	row[6] = color_diff;
	
	return row.template ext<array_type::size()>();
    }

    template <class C_> __device__ __forceinline__ array_type
    color_moment(const point_type& x,
		 const point2_type& uv, const mat4x<C_, 1>& color_diff) const
    {
	if (square(color_diff) > _sqcolor_thresh)
	    return {0};
	
	const auto	eH = _edgeH(uv.x, uv.y);
	const auto	eV = _edgeV(uv.x, uv.y);
	auto		a  = _intrinsics.image_derivative0(x, eH.x, eV.x);

	array<value_type, DOF+1>	row;
	row[0] = a.x;
	row[1] = a.y;
	row[2] = a.z;
	auto	x_cross_a = cross(x, a);
	row[3] = x_cross_a.x;
	row[4] = x_cross_a.y;
	row[5] = x_cross_a.z;
	row[6] = color_diff.x;
	auto	m = row.template ext<array_type::size()>();

	a = _intrinsics.image_derivative0(x, eH.y, eV.y);
	row[0] = a.x;
	row[1] = a.y;
	row[2] = a.z;
	x_cross_a = cross(x, a);
	row[3] = x_cross_a.x;
	row[4] = x_cross_a.y;
	row[5] = x_cross_a.z;
	row[6] = color_diff.y;
	m += row.template ext<array_type::size()>();

	a = _intrinsics.image_derivative0(x, eH.z, eV.z);
	row[0] = a.x;
	row[1] = a.y;
	row[2] = a.z;
	x_cross_a = cross(x, a);
	row[3] = x_cross_a.x;
	row[4] = x_cross_a.y;
	row[5] = x_cross_a.z;
	row[6] = color_diff.z;
	m += row.template ext<array_type::size()>();
	
	return m;
    }

  private:
    const transform_type	_Tts;
    const intrinsics_type	_intrinsics;

    const points_type		_xs;
    const directions_type	_ns;
    const image_type		_image_s;

    const points_type		_xt;
    const directions_type	_nt;
    const Texture<color_type>	_image_t;
    const Texture<color_type>	_edgeH;
    const Texture<color_type>	_edgeV;

    const value_type		_sqdist_thresh;
    const value_type		_sqangle_thresh;
    const value_type		_sqcolor_thresh;
    const value_type		_color_weight;
};
}	// namespace icp

/************************************************************************
*  class ICP<T, C, WITH_DISTORTION, CLOCK>				*
************************************************************************/
template <class T, class C, bool WITH_DISTORTION=false, class CLOCK=void>
class ICP : public Profiler<CLOCK>
{
  public:
    constexpr static size_t	NLEVELS_MAX = 5;

    using value_type		= T;
    using color_type		= C;
    using transform_type	= Rigidity<value_type, 3>;
    using intrinsics_type	= Intrinsics<value_type, WITH_DISTORTION>;
    using point_type		= typename transform_type::point_type;
    using direction_type	= typename transform_type::direction_type;

    struct Parameters
    {
	value_type	dist_thresh	= 0.1;
	value_type	angle_thresh	= 20.0;
	value_type	color_thresh	= 20.0;
	value_type	color_weight	= 1.0;
	size_t		niterations	= 5;

	friend std::ostream&
	operator <<(std::ostream& out, const Parameters& params)
	{
	    return out << "dist_thresh="  << params.dist_thresh
		       << "angle_thresh=" << params.angle_thresh
		       << "color_thresh=" << params.color_thresh
		       << "color_weight=" << params.color_weight
		       << "niterations="  << params.niterations;
	}
    };

    struct Frame
    {
	intrinsics_type		intrinsics;
	Array2<point_type>	points;
	Array2<direction_type>	normals;
	Array2<color_type>	image;
	Array2<color_type>	edgeH;
	Array2<color_type>	edgeV;

	size_t	nrow()	const	{ return points.nrow(); }
	size_t	ncol()	const	{ return points.ncol(); }
	void	swap(Frame& frame)
		{
		    std::swap(intrinsics, frame.intrinsics);
		    points.swap(frame.points);
		    normals.swap(frame.normals);
		    image.swap(frame.image);
		    edgeH.swap(frame.edgeH);
		    edgeV.swap(frame.edgeV);
		}
	void	clear()
		{
		    points.resize(0, 0);
		    normals.resize(0, 0);
		    image.resize(0, 0);
		    edgeH.resize(0, 0);
		    edgeV.resize(0, 0);
		}
    };

  private:
    using profiler_type		= Profiler<CLOCK>;

  public:
		ICP()	:profiler_type(2), _params(), _source()		{}

    const Parameters&
		getParameters()			const	{ return _params; }
    void	setParameters(const Parameters& params)	{ _params = params; }
    const Frame&
		getSourceFrame()		const	{ return _source; }
    void	setSourceFrame(const Frame& source)	;
    void	setSourceFrame(Frame&& source)		;
    void	swapSourceFrame(Frame& source)		;
    void	clearSourceFrame()			;
    bool	empty()				const	;

    value_type	operator ()(const Frame& target, transform_type& Tts)	const;
    value_type	operator ()(const Frame& target,
			    const Frame& source, transform_type& Tts)	;

  private:
    Parameters	_params;
    Frame	_source;
};

template <class T, class C, bool WITH_DISTORTION, class CLOCK> void
ICP<T, C, WITH_DISTORTION, CLOCK>::setSourceFrame(const Frame& source)
{
    _source = source;
}

template <class T, class C, bool WITH_DISTORTION, class CLOCK> void
ICP<T, C, WITH_DISTORTION, CLOCK>::setSourceFrame(Frame&& source)
{
    _source = std::move(source);
}

template <class T, class C, bool WITH_DISTORTION, class CLOCK> void
ICP<T, C, WITH_DISTORTION, CLOCK>::swapSourceFrame(Frame& source)
{
    _source.swap(source);
}

template <class T, class C, bool WITH_DISTORTION, class CLOCK> void
ICP<T, C, WITH_DISTORTION, CLOCK>::clearSourceFrame()
{
    _source.clear();
}

template <class T, class C, bool WITH_DISTORTION, class CLOCK> bool
ICP<T, C, WITH_DISTORTION, CLOCK>::empty() const
{
    return _source.nrow() == 0;
}

template <class T, class C, bool WITH_DISTORTION, class CLOCK>
typename ICP<T, C, WITH_DISTORTION, CLOCK>::value_type
ICP<T, C, WITH_DISTORTION, CLOCK>::operator ()(const Frame& target,
					       transform_type& Tts) const
{
  // Update transform by Lebensberg-Marquarde iteration.
    auto	Tts_old = Tts;
    auto	mse_old = std::numeric_limits<value_type>::max();
    value_type	lambda	= 1.0e-3;

    for (size_t n = 0; n < _params.niterations; ++n)
    {
	using error_metric_type	= icp::ErrorMetric<ICP>;
	using matrix_type	= typename error_metric_type::matrix_type;
	using vector_type	= typename error_metric_type::vector_type;
	using array_type	= typename error_metric_type::array_type;
    
      // Compute point moment by parallel reduction.
	constexpr static value_type	COLOR_WEIGHT_SCALE = 1.0e-6;
	
	error_metric_type	error_metric(Tts, _source, target,
					     _params.dist_thresh,
					     _params.angle_thresh,
					     _params.color_thresh,
					     COLOR_WEIGHT_SCALE *
					     _params.color_weight);
	Array<array_type>	tmp_error_metric(1);
	size_t			tmp_size = 0;
	cub::DeviceReduce::Sum(nullptr, tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   error_metric),
			       tmp_error_metric.begin(), error_metric.size());
	Array<uint8_t>		tmp(tmp_size);
	cub::DeviceReduce::Sum(tmp.data().get(), tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   error_metric),
			       tmp_error_metric.begin(), error_metric.size());
	gpuCheckLastError();
	const auto		error_metric_array = tmp_error_metric[0];

      // Evaluate residula mean square errors in point and color.
	const auto	mse = error_metric_type::mse(error_metric_array);
	if (mse < mse_old)
	{
	    constexpr static value_type	tol = 1.0e-5;
	    if (std::abs(mse - mse_old) <= tol*(mse + mse_old + 1.0e-10))
		return mse;

	    Tts_old = Tts;
	    mse_old = mse;
	    lambda *= 0.1;
	}
	else
	{
	    if (lambda < 1.0e-10)
	    {
		Tts = Tts_old;
		return mse_old;
	    }

	    lambda *= 10.0;
	}

      // Solve the linear system for updates of transform.
	matrix_type		A = error_metric_type::M(error_metric_array);
	for (size_t i = 0; i < A.rows(); ++i)
	    A(i, i) *= (1.0 + lambda);
	const vector_type	b = error_metric_type::d(error_metric_array);
	const auto		update = A.ldlt().solve(b).eval();
	Tts = transform_type::exp(update.data()) * Tts_old;
#if !defined(NDEBUG)
	std::cerr << "  [" << n
		  << "]: err=" << std::sqrt(mse)
		  << ", lambda=" << lambda
		  << std::endl;
#endif
    }

    Tts = Tts_old;
    return mse_old;
}

template <class T, class C, bool WITH_DISTORTION, class CLOCK>
typename ICP<T, C, WITH_DISTORTION, CLOCK>::value_type
ICP<T, C, WITH_DISTORTION, CLOCK>::operator ()(
    const Frame& source, const Frame& target, transform_type& Tts)
{
    setSourceFrame(source);
    return (*this)(target, Tts);
}
}	// namespace TU::cu
