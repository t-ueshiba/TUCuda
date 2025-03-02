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
*  class PointPlaneError<ICP>						*
************************************************************************/
template <class ICP>
class PointPlaneError
{
  public:
    using value_type		= typename ICP::value_type;
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
    using points_type		= range<range_iterator<
					    thrust::device_ptr<
						const point_type> > >;
    using directions_type	= range<range_iterator<
					    thrust::device_ptr<
						const direction_type> > >;

  public:
    PointPlaneError(const transform_type& Tts,
		    const frame_type& source, const frame_type& target,
		    value_type dist_thresh, value_type angle_thresh)
	:_Tts(Tts), _intrinsics(target.intrinsics),
	 _xs(source.points.cbegin(),  source.points.nrow()),
	 _ns(source.normals.cbegin(), source.normals.nrow()),
	 _xt(target.points.cbegin(),  target.points.nrow()),
	 _nt(target.normals.cbegin(), target.normals.nrow()),
	 _sqdist_thresh(dist_thresh*dist_thresh),
	 _sqangle_thresh(angle_thresh*angle_thresh)
    {
    }

    __device__ __forceinline__ array_type
    operator()(int i) const
    {
	const int		v    = i / ncol();
	const int		u    = i - (v * ncol());
	const point_type	xs   = _xs[v][u];
	const auto		xs_t = _Tts(xs);
	const auto		uv_t = _intrinsics(xs_t);
	const int		ut   = device::to_int(uv_t.x);
	const int		vt   = device::to_int(uv_t.y);

	if (xs.z > 0 && xs_t.z > 0 &&
	    0 <= ut && ut < ncol() && 0 <= vt && vt < nrow())
	{
	    const point_type		xt   = _xt[vt][ut];
	    const direction_type	nt   = _nt[vt][ut];
	    const direction_type	ns   = _ns[v][u];
	    const direction_type	ns_t = _Tts.direction(ns);

	    if (nt.z > 0 && ns.z > 0 &&
		square(cross(ns_t, nt)) < _sqangle_thresh &&
		square(xt - xs_t)	< _sqdist_thresh)
	    {
		array<value_type, DOF+1>	row;
		row[0] = nt.x;
		row[1] = nt.y;
		row[2] = nt.z;
		const auto	x_cross_n = cross(xs_t, nt);
		row[3] = x_cross_n.x;
		row[4] = x_cross_n.y;
		row[5] = x_cross_n.z;
		row[6] = dot(nt, xt - xs_t);	// deviation term

		auto	m = row.template ext<array_type::size()>();
		m[array_type::size()-1] = 1;	// npoints term

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

  private:
    const transform_type	_Tts;
    const intrinsics_type	_intrinsics;

    const points_type		_xs;
    const directions_type	_ns;

    const points_type		_xt;
    const directions_type	_nt;

    const value_type		_sqdist_thresh;
    const value_type		_sqangle_thresh;
};

/************************************************************************
*  class ColorError<ICP>						*
************************************************************************/
template <class ICP>
class ColorError
{
  public:
    using value_type		= typename ICP::value_type;
    using color_type		= typename ICP::color_type;
    using transform_type	= typename ICP::transform_type;
    using intrinsics_type	= typename ICP::intrinsics_type;
    using frame_type		= typename ICP::Frame;

    constexpr static size_t	DOF = transform_type::DOF;

    using array_type		= array<value_type, DOF*(DOF+1)/2 + 1>;
    using matrix_type		= Eigen::Matrix<value_type, DOF, DOF>;

  private:
    using param_type		= typename transform_type::param_type;
    using point_type		= typename transform_type::point_type;
    using points_type		= range<range_iterator<
					    thrust::device_ptr<
						const point_type> > >;
    using image_type		= range<range_iterator<
					    thrust::device_ptr<
						const color_type> > >;

  public:
    ColorError(const transform_type& Tts,
	       const frame_type& source, const frame_type& target)
	:_Tts(Tts), _intrinsics(target.intrinsics),
	 _xs(source.points.cbegin(), source.points.nrow()),
	 _xt(target.points.cbegin(), target.points.nrow()),
	 _image_s(source.image.cbegin(), source.image.nrow()),
	 _image_t(target.image),
	 _edgeH(target.edgeH),
	 _edgeV(target.edgeV)
    {
    }

    template <class C=color_type> __device__ __forceinline__
    std::enable_if_t<std::is_arithmetic<C>::value, array_type>
    operator ()(int i) const
    {
	const int		v    = i / ncol();
	const int		u    = i - (v * ncol());
	const point_type	xs   = _xs[v][u];
	const point_type	xs_t = _Tts(xs);
	const auto		uv_t = _intrinsics(xs_t);
	const int		ut   = device::to_int(uv_t.x);
	const int		vt   = device::to_int(uv_t.y);

	if (xs.z > 0 && xs_t.z > 0 &&
	    0 <= ut && ut < ncol() && 0 <= vt && vt < nrow())
	{
	    const C	eH = _edgeH(uv_t.x, uv_t.y);
	    const C	eV = _edgeV(uv_t.x, uv_t.y);
	    const auto	a  = _intrinsics.image_derivative0(xs_t, eH, eV);

	    array<value_type, DOF+1>	row;
	    row[0] = a.x;
	    row[1] = a.y;
	    row[2] = a.z;
	    const auto	x_cross_a = cross(xs_t, a);
	    row[3] = x_cross_a.x;
	    row[4] = x_cross_a.y;
	    row[5] = x_cross_a.z;
	    row[6] = _image_s[v][u] - _image_t(uv_t.x, uv_t.y);

	    auto	m = row.template ext<array_type::size()>();
	    m[array_type::size()-1] = 1;
	    
	    return m;
	}

	return {0};
    }

    template <class C=color_type> __device__ __forceinline__
    std::enable_if_t<!std::is_arithmetic<C>::value, array_type>
    operator ()(int i) const
    {
	const int		v    = i / ncol();
	const int		u    = i - (v * ncol());
	const point_type	xs   = _xs[v][u];
	const point_type	xs_t = _Tts(xs);
	const auto		uv_t = _intrinsics(xs_t);
	const int		ut   = device::to_int(uv_t.x);
	const int		vt   = device::to_int(uv_t.y);

	if (xs.z > 0 && xs_t.z > 0 &&
	    0 <= ut && ut < ncol() && 0 <= vt && vt < nrow())
	{
	    const C	eH = _edgeH[vt][ut];
	    const C	eV = _edgeV[vt][ut];
	    auto	a  = _intrinsics.image_derivative0(xs_t, eH.x, eV.x);
	    param_type	row;
	    row[0] = a.x;
	    row[1] = a.y;
	    row[2] = a.z;
	    auto	x_cross_a = cross(xs_t, a);
	    row[3] = x_cross_a.x;
	    row[4] = x_cross_a.y;
	    row[5] = x_cross_a.z;
	    auto	m = row.template ext();

	    a = _intrinsics.image_derivative0(xs_t, eH.y, eV.y);
	    row[0] = a.x;
	    row[1] = a.y;
	    row[2] = a.z;
	    x_cross_a = cross(xs_t, a);
	    row[3] = x_cross_a.x;
	    row[4] = x_cross_a.y;
	    row[5] = x_cross_a.z;
	    m += row.template ext();

	    a = _intrinsics.image_derivative0(xs_t, eH.z, eV.z);
	    row[0] = a.x;
	    row[1] = a.y;
	    row[2] = a.z;
	    x_cross_a = cross(xs_t, a);
	    row[3] = x_cross_a.x;
	    row[4] = x_cross_a.y;
	    row[5] = x_cross_a.z;
	    m += row.template ext();

	    return m;
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
	auto			p = moment.data();
	for (int i = 0; i < m.rows(); ++i)
	    for (int j = i; j < m.cols(); ++j)
		m(j, i) = m(i, j) = *p++;
	return m;
    }

  private:
    __host__ __device__ __forceinline__
    int		nrow()		const	{ return _edgeH.size(); }
    __host__ __device__ __forceinline__
    int		ncol()		const	{ return _edgeH.cbegin().size(); }

  private:
    const transform_type	_Tts;
    const intrinsics_type	_intrinsics;
    const points_type		_xs;
    const points_type		_xt;
    const image_type		_image_s;
    const Texture<color_type>	_image_t;
    const Texture<color_type>	_edgeH;
    const Texture<color_type>	_edgeV;
};

/************************************************************************
*  class ColorDeviation<ICP>						*
************************************************************************/
template <class ICP>
class ColorDeviation
{
  public:
    using value_type		= typename ICP::value_type;
    using color_type		= typename ICP::color_type;
    using transform_type	= typename ICP::transform_type;
    using intrinsics_type	= typename ICP::intrinsics_type;
    using frame_type		= typename ICP::Frame;

    constexpr static size_t	DOF = transform_type::DOF;

    using array_type		= array<value_type, DOF+2>;
    using vector_type		= Eigen::Matrix<value_type, DOF, 1>;

  private:
    using param_type		= typename transform_type::param_type;
    using point_type		= typename transform_type::point_type;
    using points_type		= range<range_iterator<
					    thrust::device_ptr<
						const point_type> > >;
    using image_type		= range<range_iterator<
					    thrust::device_ptr<
						const color_type> > >;

  public:
    ColorDeviation(const transform_type& Tts,
		   const Array2<color_type>& image_s,
		   const frame_type& target, value_type color_thresh)
	:_Tst(Tts.inv()),
	 _intrinsics(target.intrinsics),
	 _image_s(image_s),
	 _points_t(target.points.cbegin(), target.points.nrow()),
	 _image_t(target.image.cbegin(), target.image.nrow()),
	 _edgeH(target.edgeH.cbegin(), target.edgeH.nrow()),
	 _edgeV(target.edgeV.cbegin(), target.edgeV.nrow()),
	 _sqcolor_thresh(color_thresh*color_thresh)
    {
    }

    template <class C=color_type> __device__ __forceinline__
    std::enable_if_t<std::is_arithmetic<C>::value, array_type>
    operator ()(int i) const
    {
	const int		v  = i / ncol();
	const int		u  = i - (v * ncol());
	const point_type	xt = _points_t[v][u];

	if (xt.z > 0)
	{
	  // Project a source point transfered to the destination pose.
	    const auto	uv_s = _intrinsics(_Tst(xt));

	    if (0 <= uv_s.x && uv_s.x < ncol() &&
		0 <= uv_s.y && uv_s.y < nrow())
	    {
		const auto	b = _image_s(uv_s.x, uv_s.y) - _image_t[v][u];

		if (b*b < _sqcolor_thresh)
		{
		    const auto	ab = _intrinsics.image_derivative0(
					xt, _edgeH[v][u], _edgeV[v][u])
				   * b;
		    param_type	row;
		    row[0] = ab.x;
		    row[1] = ab.y;
		    row[2] = ab.z;
		    const auto	x_cross_ab = cross(xt, ab);
		    row[3] = x_cross_ab.x;
		    row[4] = x_cross_ab.y;
		    row[5] = x_cross_ab.z;
		    auto	d = row.template extend<DOF+2>();
		    d[DOF]   = b*b;
		    d[DOF+1] = 1;

		    return d;
		}
	    }
	}

	return {0};
    }

    template <class C=color_type> __device__ __forceinline__
    std::enable_if_t<!std::is_arithmetic<C>::value, array_type>
    operator ()(int i) const
    {
	const int		v  = i / ncol();
	const int		u  = i - (v * ncol());
	const point_type	xt = _points_t[v][u];

	if (xt.z > 0)
	{
	    const auto	uv_s = _intrinsics(_Tst(xt));

	    if (0 <= uv_s.x && uv_s.x < ncol() &&
		0 <= uv_s.y && uv_s.y < nrow())
	    {
		const auto	b = _image_s(uv_s.x, uv_s.y) - _image_t[v][u];

		if (square(b) < _sqcolor_thresh)
		{
		    const C	eH = _edgeH[v][u];
		    const C	eV = _edgeV[v][u];
		    const auto	ab = _intrinsics.image_derivative0(xt,
								   eH.x, eV.x)
				   * b.x
				   + _intrinsics.image_derivative0(xt,
								   eH.y, eV.y)
				   * b.y
				   + _intrinsics.image_derivative0(xt,
								   eH.z, eV.z)
				   * b.z;
		    param_type	row;
		    row[0] = ab.x;
		    row[1] = ab.y;
		    row[2] = ab.z;
		    const auto	x_cross_ab = cross(xt, ab);
		    row[3] = x_cross_ab.x;
		    row[4] = x_cross_ab.y;
		    row[5] = x_cross_ab.z;
		    auto	d = row.template extend<DOF+2>();
		    d[DOF]   = square(b);
		    d[DOF+1] = 1;

		    return d;
		}
	    }
	}

	return {0};
    }

    int
    size() const
    {
	return nrow() * ncol();
    }

    static vector_type
    d(const array_type& deviation)
    {
	vector_type	v;
	v << deviation[0], deviation[1], deviation[2],
	     deviation[3], deviation[4], deviation[5];

	return v;
    }

    static value_type
    npoints(const array_type& deviation)
    {
	return deviation[DOF+1];
    }

    static value_type
    mse(const array_type& deviation)
    {
	return deviation[DOF] / deviation[DOF+1];
    }

  private:
    __host__ __device__ __forceinline__
    int		nrow()		const	{ return _image_t.size(); }
    __host__ __device__ __forceinline__
    int		ncol()		const	{ return _image_t.cbegin().size(); }

  private:
    const transform_type	_Tst;
    const intrinsics_type	_intrinsics;
    const Texture<color_type>	_image_s;
    const points_type		_points_t;
    const image_type		_image_t;
    const image_type		_edgeH;
    const image_type		_edgeV;
    const value_type		_sqcolor_thresh;
};
}	// namespace icp

/************************************************************************
*  class ICP<T, C, WD, CLOCK>						*
************************************************************************/
template <class T, class C, bool WD=false, class CLOCK=void>
class ICP : public Profiler<CLOCK>
{
  public:
    constexpr static size_t	NLEVELS_MAX = 5;
    constexpr static bool	with_distortion = WD;

    using value_type		= T;
    using color_type		= C;
    using transform_type	= Rigidity<value_type, 3>;
    using intrinsics_type	= Intrinsics<value_type, with_distortion>;
    using point_type		= typename transform_type::point_type;
    using direction_type	= typename transform_type::direction_type;

    struct Parameters
    {
	value_type	dist_thresh	= 0.1;
	value_type	angle_thresh	= 20.0;
	value_type	color_thresh	= 20.0;
	value_type	color_weight	= 0.1;
	size_t		niterations	= 5;
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

    constexpr static value_type	COLOR_WEIGHT_SCALE = 1.0e-6;
};

template <class T, class C, bool WD, class CLOCK> void
ICP<T, C, WD, CLOCK>::setSourceFrame(const Frame& source)
{
    _source = source;
}

template <class T, class C, bool WD, class CLOCK> void
ICP<T, C, WD, CLOCK>::setSourceFrame(Frame&& source)
{
    _source = std::move(source);
}

template <class T, class C, bool WD, class CLOCK> void
ICP<T, C, WD, CLOCK>::swapSourceFrame(Frame& source)
{
    _source.swap(source);
}

template <class T, class C, bool WD, class CLOCK> void
ICP<T, C, WD, CLOCK>::clearSourceFrame()
{
    _source.clear();
}

template <class T, class C, bool WD, class CLOCK> bool
ICP<T, C, WD, CLOCK>::empty() const
{
    return _source.nrow() == 0;
}

template <class T, class C, bool WD, class CLOCK>
typename ICP<T, C, WD, CLOCK>::value_type
ICP<T, C, WD, CLOCK>::operator ()(const Frame& target,
				  transform_type& Tts) const
{
    using point_error_type	     = icp::PointPlaneError<ICP>;
    using color_moment_type	     = icp::ColorMoment<ICP>;
    using color_deviation_type	     = icp::ColorDeviation<ICP>;
    using matrix_type		     = typename point_error_type::matrix_type;
    using vector_type		     = typename point_error_type::vector_type;
    using point_error_array_type     = typename point_error_type::array_type;
    using color_moment_array_type    = typename color_moment_type::array_type;
    using color_deviation_array_type = typename color_deviation_type::array_type;

  // Compute color moment of the target frame by parallel reduction.
    const color_moment_type		color_moment(target);
    Array<color_moment_array_type>	tmp_color_moment(1);
    size_t				tmp_size = 0;
    cub::DeviceReduce::Sum(nullptr, tmp_size,
			   thrust::make_transform_iterator(
			       thrust::make_counting_iterator(0),
			       color_moment),
			   tmp_color_moment.begin(), color_moment.size());
    Array<uint8_t>	tmp(tmp_size);
    cub::DeviceReduce::Sum(tmp.data().get(), tmp_size,
			   thrust::make_transform_iterator(
			       thrust::make_counting_iterator(0),
			       color_moment),
			   tmp_color_moment.begin(), color_moment.size());
    gpuCheckLastError();
    const auto	color_moment_array = tmp_color_moment[0];

  // Update transform by Lebensberg-Marquarde iteration.
    auto	Tts_old = Tts;
    auto	mse_old = std::numeric_limits<value_type>::max();
    value_type	lambda	= 1.0e-3;

    for (size_t n = 0; n < _params.niterations; ++n)
    {
      // Compute point moment by parallel reduction.
	const point_error_type		point_error(Tts, _source, target,
						    _params.dist_thresh,
						    _params.angle_thresh);
	Array<point_error_array_type>	tmp_point_error(1);
	size_t				tmp_size = 0;
	cub::DeviceReduce::Sum(nullptr, tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   point_error),
			       tmp_point_error.begin(), point_error.size());
	Array<uint8_t>	tmp(tmp_size);
	cub::DeviceReduce::Sum(tmp.data().get(), tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   point_error),
			       tmp_point_error.begin(), point_error.size());
	gpuCheckLastError();
	const auto	point_error_array = tmp_point_error[0];

      // Compute color deviation by parallel reduction.
	const color_deviation_type	color_deviation(Tts,
							_source.image, target,
							_params.color_thresh);
	Array<color_deviation_array_type>	tmp_color_deviation(1);
	tmp_size = 0;
	cub::DeviceReduce::Sum(nullptr, tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   color_deviation),
			       tmp_color_deviation.begin(),
			       color_deviation.size());
	if (tmp_size > tmp.size())
	    tmp.resize(tmp_size);
	cub::DeviceReduce::Sum(tmp.data().get(), tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   color_deviation),
			       tmp_color_deviation.begin(),
			       color_deviation.size());
	gpuCheckLastError();
	const auto	color_deviation_array = tmp_color_deviation[0];

      // Evaluate residula mean square errors in point and color.
	const auto	point_mse = point_error_type::mse(point_error_array);
	const auto	color_mse = color_deviation_type
					::mse(color_deviation_array);
	const auto	mse = point_mse
			    + COLOR_WEIGHT_SCALE*_params.color_weight
			    * color_mse;
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
	matrix_type	A = point_error_type::M(point_error_array)
			  + COLOR_WEIGHT_SCALE*_params.color_weight
			  * color_moment_type::M(color_moment_array);
	for (size_t i = 0; i < A.rows(); ++i)
	    A(i, i) *= (1.0 + lambda);
	const vector_type b = point_error_type::d(point_error_array)
		      	    + COLOR_WEIGHT_SCALE*_params.color_weight
			    * color_deviation_type::d(color_deviation_array);
	const auto	  update = A.ldlt().solve(b).eval();
	Tts = transform_type::exp(update.data()) * Tts_old;
#if !defined(NDEBUG)
	// std::cerr << "--- A ---\n" << point_error_type::M(point_moment)
	// 	  << std::endl;
	// std::cerr << "--- b ---\n" << point_error_type::d(point_moment)
	// 	  << std::endl;
	// std::cerr << "--- C ---\n" << color_moment_metric_type::M(color_moment)
	// 	  << std::endl;
	// std::cerr << "--- d ---\n"
	// 	  << color_deviation_metric_type::d(color_deviation)
	// 	  << std::endl;
	// std::cerr << "--- A_C ---\n" << A << std::endl;
	// std::cerr << "--- b_d ---\n" << b << std::endl;
	// std::cerr << "--- update ---\n" << update << std::endl;

	std::cerr << "  [" << n
		  << "]: point_err=" << std::sqrt(point_mse)
		  << ", color_err=" << std::sqrt(color_mse)
		  << ", lambda=" << lambda
		  << std::endl;
#endif
    }

    Tts = Tts_old;
    return mse_old;
}

template <class T, class C, bool WD, class CLOCK>
typename ICP<T, C, WD, CLOCK>::value_type
ICP<T, C, WD, CLOCK>::operator ()(const Frame& source,
				  const Frame& target, transform_type& Tts)
{
    setSourceFrame(source);
    return (*this)(target, Tts);
}
}	// namespace TU::cu
