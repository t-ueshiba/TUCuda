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
  \file		vec.h
  \author	Toshio UESHIBA
  \brief	cudaベクトルクラスとその演算子の定義
*/
#pragma once

#include <cstdint>
#include <thrust/device_ptr.h>
#include "TU/Image++.h"		// for TU::RGB_<E>

namespace TU
{
namespace cuda
{
#if defined(__NVCC__)
namespace device
{
  template <class T>
  constexpr static T	minval  = std::numeric_limits<T>::min();
  template <class T>
  constexpr static T	maxval  = std::numeric_limits<T>::max();
  template <class T>
  constexpr static T	epsilon = std::numeric_limits<T>::epsilon();
    
  __device__ inline float  fma(float x,
			       float y, float z)	{ return fmaf(x, y, z);}
  __device__ inline float  min(float x, float y)	{ return fminf(x, y); }
  __device__ inline float  max(float x, float y)	{ return fmaxf(x, y); }
  __device__ inline float  square(float x)		{ return x*x; }
  __device__ inline float  abs(float x)			{ return fabsf(x); }
  __device__ inline float  sqrt(float x)		{ return sqrtf(x); }
  __device__ inline float  rsqrt(float x)		{ return rsqrtf(x); }
  __device__ inline float  sin(float x)			{ return sinf(x); }
  __device__ inline float  cos(float x)			{ return cosf(x); }
  __device__ inline float  atan2(float y, float x)	{ return atan2f(y, x); }
  __device__ inline float  floor(float x)		{ return floorf(x); }
  __device__ inline float  ceil(float x)		{ return ceilf(x); }

  __device__ inline double min(double x, double y)	{ return fmin(x, y); }
  __device__ inline double max(double x, double y)	{ return fmax(x, y); }
  __device__ inline double square(double x)		{ return x*x; }
  __device__ inline double abs(double x)		{ return fabs(x); }
}	// namespace device
#endif	// __NVCC__

namespace detail
{
  template <class T, size_t N>	struct base_vec;

  template <>	struct base_vec<int8_t,   1>	{ using type = char1;	};
  template <>	struct base_vec<int8_t,   2>	{ using type = char2;	};
  template <>	struct base_vec<int8_t,   3>	{ using type = char3;	};
  template <>	struct base_vec<int8_t,   4>	{ using type = char4;	};

  template <>	struct base_vec<uint8_t,  1>	{ using type = uchar1;	};
  template <>	struct base_vec<uint8_t,  2>	{ using type = uchar2;	};
  template <>	struct base_vec<uint8_t,  3>	{ using type = uchar3;	};
  template <>	struct base_vec<uint8_t,  4>	{ using type = uchar4;	};

  template <>	struct base_vec<int16_t,  1>	{ using type = short1;	};
  template <>	struct base_vec<int16_t,  2>	{ using type = short2;	};
  template <>	struct base_vec<int16_t,  3>	{ using type = short3;	};
  template <>	struct base_vec<int16_t,  4>	{ using type = short4;	};

  template <>	struct base_vec<uint16_t, 1>	{ using type = ushort1;	};
  template <>	struct base_vec<uint16_t, 2>	{ using type = ushort2;	};
  template <>	struct base_vec<uint16_t, 3>	{ using type = ushort3;	};
  template <>	struct base_vec<uint16_t, 4>	{ using type = ushort4;	};

  template <>	struct base_vec<int32_t,  1>	{ using type = int1;	};
  template <>	struct base_vec<int32_t,  2>	{ using type = int2;	};
  template <>	struct base_vec<int32_t,  3>	{ using type = int3;	};
  template <>	struct base_vec<int32_t,  4>	{ using type = int4;	};

  template <>	struct base_vec<uint32_t, 1>	{ using type = uint1;	};
  template <>	struct base_vec<uint32_t, 2>	{ using type = uint2;	};
  template <>	struct base_vec<uint32_t, 3>	{ using type = uint3;	};
  template <>	struct base_vec<uint32_t, 4>	{ using type = uint4;	};

  template <>	struct base_vec<float,    1>	{ using type = float1;	};
  template <>	struct base_vec<float,    2>	{ using type = float2;	};
  template <>	struct base_vec<float,    3>	{ using type = float3;	};
  template <>	struct base_vec<float,    4>	{ using type = float4;	};

  template <>	struct base_vec<int64_t,  1>	{ using type = longlong1; };
  template <>	struct base_vec<int64_t,  2>	{ using type = longlong2; };

  template <>	struct base_vec<double,   1>	{ using type = double1;	};
  template <>	struct base_vec<double,   2>	{ using type = double2;	};
}	// namespace derail

/************************************************************************
*  struct mat2x<T, C>, mat3x<T, C>, mat4x<T, C>				*
************************************************************************/
template <class T, size_t C>	struct mat2x;

template <class T>
struct mat2x<T, 1> : public detail::base_vec<T, 2>::type
{
    using super		= typename detail::base_vec<T, 2>::type;

    using element_type	= T;
    using value_type	= element_type;

    using super::x;
    using super::y;

    __host__ __device__ constexpr
    static size_t	rank()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    __host__ __device__	mat2x()			:super()		{}
    __host__ __device__ constexpr
			mat2x(T x, T y=0)	:super{x, y}		{}

    __host__ __device__
    mat2x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    return *this;
			}
};
    
template <class T, size_t C>	struct mat3x;

template <class T>
struct mat3x<T, 1> : public detail::base_vec<T, 3>::type
{
    using super		= typename detail::base_vec<T, 3>::type;

    using element_type	= T;
    using value_type	= element_type;

    using super::x;
    using super::y;
    using super::z;

    __host__ __device__ constexpr
    static size_t	rank()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 3; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    __host__ __device__	mat3x()			 :super()		{}
    __host__ __device__ constexpr
			mat3x(T x, T y=0, T z=0) :super{x, y, z}	{}

    __host__ __device__
    mat3x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    z = c;
			    return *this;
			}
};
    
template <class T, size_t C>	struct mat4x;

template <class T>
struct mat4x<T, 1> : public detail::base_vec<T, 4>::type
{
    using super		= typename detail::base_vec<T, 4>::type;

    using element_type	= T;
    using value_type	= element_type;

    using super::x;
    using super::y;
    using super::z;
    using super::w;

    __host__ __device__ constexpr
    static size_t	rank()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 4; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    __host__ __device__	mat4x()			   :super()		{}
    __host__ __device__ constexpr
			mat4x(T x,
			      T y=0, T z=0, T w=0) :super{x, y, z, w}	{}

    __host__ __device__
    mat4x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    z = c;
			    w = c;
			    return *this;
			}
};

template <class T, size_t D>
using vec = std::conditional_t<D == 1, T,
	    std::conditional_t<D == 2, mat2x<T, 1>,
	    std::conditional_t<D == 3, mat3x<T, 1>,
	    std::conditional_t<D == 4, mat4x<T, 1>, void> > > >;

template <class T, size_t C>
struct mat2x
{
    using element_type		= T;
    using value_type		= vec<T, C>;

    __host__ __device__ constexpr
    static size_t	rank()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return C; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 2, mat2x<T, 2> >
    transpose()	const
    {
	return {{x.x, y.x}, {x.y, y.y}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 3, mat3x<T, 2> >
    transpose()	const
    {
	return {{x.x, y.x}, {x.y, y.y}, {x.z, y.z}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 4, mat4x<T, 2> >
    transpose()	const
    {
	return {{x.x, y.x}, {x.y, y.y}, {x.z, y.z}, {x.w, y.w}};
    }

    __host__ __device__ mat2x&
    operator =(T c)
    {
	x = c;
	y = c;
	return *this;
    }

    value_type	x, y;
};

template <class T, size_t C>
struct mat3x
{
    using element_type		= T;
    using value_type		= vec<T, C>;

    __host__ __device__ constexpr
    static size_t	rank()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 3; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return C; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 2, mat2x<T, 3> >
    transpose()	const
    {
	return {{x.x, y.x, z.x}, {x.y, y.y, z.y}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 3, mat3x<T, 3> >
    transpose()	const
    {
	return {{x.x, y.x, z.x}, {x.y, y.y, z.y}, {x.z, y.z, z.z}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 4, mat4x<T, 3> >
    transpose()	const
    {
	return {{x.x, y.x, z.x}, {x.y, y.y, z.y},
		{x.z, y.z, z.z}, {x.w, y.w, z.w}};
    }

    __host__ __device__ mat3x&
    operator =(T c)
    {
	x = c;
	y = c;
	z = c;
	return *this;
    }

    value_type	x, y, z;
};

template <class T, size_t C>
struct mat4x
{
    using element_type		= T;
    using value_type		= vec<T, C>;

    __host__ __device__ constexpr
    static size_t	rank()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 4; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return C; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 2, mat2x<T, 4> >
    transpose()	const
    {
	return {{x.x, y.x, z.x, w.x}, {x.y, y.y, z.y, w.y}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 3, mat3x<T, 4> >
    transpose()	const
    {
	return {{x.x, y.x, z.x, w.x},
		{x.y, y.y, z.y, w.y}, {x.z, y.z, z.z, w.z}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 4, mat4x<T, 4> >
    transpose()	const
    {
	return {{x.x, y.x, z.x, w.x}, {x.y, y.y, z.y, w.y},
		{x.z, y.z, z.z, w.z}, {x.w, y.w, z.w, w.w}};
    }

    __host__ __device__ mat4x&
    operator =(T c)
    {
	x = c;
	y = c;
	z = c;
	w = c;
	return *this;
    }

    value_type	x, y, z, w;
};

template <class VM> constexpr size_t
size1()
{
    return VM::size1();
}

// TU::element_t<E> requires begin(std::decval<E>()) be valid.
// We have to override TU::cuda::begin() defined in tuple.h,
template <class T, size_t C> const typename mat2x<T, C>::value_type*
begin(const mat2x<T, C>&)						;

template <class T, size_t C> const typename mat3x<T, C>::value_type*
begin(const mat3x<T, C>&)						;

template <class T, size_t C> const typename mat4x<T, C>::value_type*
begin(const mat4x<T, C>&)						;

template <class T, size_t R, size_t C>
using mat = std::conditional_t<R == 1, vec<T, C>,
	    std::conditional_t<R == 2, mat2x<T, C>,
	    std::conditional_t<R == 3, mat3x<T, C>,
	    std::conditional_t<R == 4, mat4x<T, C>, void> > > >;

/************************************************************************
*  Access element by its integral index					*
************************************************************************/
template <size_t I, class VM, std::enable_if_t<I == 0>* = nullptr>
__host__ __device__ inline auto		val(const VM& a)	{ return a.x; }
template <size_t I, class VM, std::enable_if_t<I == 1>* = nullptr>
__host__ __device__ inline auto		val(const VM& a)	{ return a.y; }
template <size_t I, class VM, std::enable_if_t<I == 2>* = nullptr>
__host__ __device__ inline auto		val(const VM& a)	{ return a.z; }
template <size_t I, class VM, std::enable_if_t<I == 3>* = nullptr>
__host__ __device__ inline auto		val(const VM& a)	{ return a.w; }

template <size_t I, class VM, std::enable_if_t<I == 0>* = nullptr>
__host__ __device__ inline auto&	val(VM& a)		{ return a.x; }
template <size_t I, class VM, std::enable_if_t<I == 1>* = nullptr>
__host__ __device__ inline auto&	val(VM& a)		{ return a.y; }
template <size_t I, class VM, std::enable_if_t<I == 2>* = nullptr>
__host__ __device__ inline auto&	val(VM& a)		{ return a.z; }
template <size_t I, class VM, std::enable_if_t<I == 3>* = nullptr>
__host__ __device__ inline auto&	val(VM& a)		{ return a.w; }

/************************************************************************
*  2-dimensional vectors or 2-by-C matrices				*
************************************************************************/
template <class T, size_t C> __host__ __device__ inline mat2x<T, C>
operator -(const mat2x<T, C>& a)
{
    return {-a.x, -a.y};
}

template <class T, size_t C> __host__ __device__ inline mat2x<T, C>&
operator +=(mat2x<T, C>& a, const mat2x<T, C>& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template <class T, size_t C> __host__ __device__ inline mat2x<T, C>&
operator -=(mat2x<T, C>& a, const mat2x<T, C>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template <class T, size_t C> __host__ __device__ inline mat2x<T, C>&
operator *=(mat2x<T, C>& a, T c)
{
    a.x *= c;
    a.y *= c;
    return a;
}

template <class T, size_t C> __host__ __device__ inline mat2x<T, C>
operator +(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return {a.x + b.x, a.y + b.y};
}

template <class T, size_t C> __host__ __device__ inline mat2x<T, C>
operator -(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return {a.x - b.x, a.y - b.y};
}

template <class T, size_t C> __host__ __device__ inline mat2x<T, C>
operator *(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return {a.x * b.x, a.y * b.y};
}

template <class T, size_t C> __host__ __device__ inline mat2x<T, C>
operator /(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return {a.x / b.x, a.y / b.y};
}

template <class T, size_t C> __host__ __device__ inline mat2x<T, C>
operator *(const mat2x<T, C>& a, T c)
{
    return {a.x * c, a.y * c};
}

template <class T, size_t C> __host__ __device__ inline mat2x<T, C>
operator /(T c, const mat2x<T, C>& a)
{
    return {c / a.x, c / a.y};
}

template <class T, size_t C> __host__ __device__ inline bool
operator ==(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return a.x == b.x && a.y == b.y;
}

template <class T, size_t C> __host__ __device__ inline bool
operator !=(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return !(a == b);
}

/************************************************************************
*  3-dimensional vectors or 3-by-C matrices				*
************************************************************************/
template <class T, size_t C> __host__ __device__ inline mat3x<T, C>
operator -(const mat3x<T, C>& a)
{
    return {-a.x, -a.y, -a.z};
}

template <class T, size_t C> __host__ __device__ inline mat3x<T, C>&
operator +=(mat3x<T, C>& a, const mat3x<T, C>& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template <class T, size_t C> __host__ __device__ inline mat3x<T, C>&
operator -=(mat3x<T, C>& a, const mat3x<T, C>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template <class T, size_t C> __host__ __device__ inline mat3x<T, C>&
operator *=(mat3x<T, C>& a, T c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}

template <class T, size_t C> __host__ __device__ inline mat3x<T, C>
operator +(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <class T, size_t C> __host__ __device__ inline mat3x<T, C>
operator -(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <class T, size_t C> __host__ __device__ inline mat3x<T, C>
operator *(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

template <class T, size_t C> __host__ __device__ inline mat3x<T, C>
operator /(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

template <class T, size_t C> __host__ __device__ inline mat3x<T, C>
operator *(const mat3x<T, C>& a, T c)
{
    return {a.x * c, a.y * c, a.z * c};
}

template <class T, size_t C> __host__ __device__ inline mat3x<T, C>
operator /(T c, const mat3x<T, C>& a)
{
    return {c / a.x, c / a.y, c / a.z};
}

template <class T, size_t C> __host__ __device__ inline bool
operator ==(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

template <class T, size_t C> __host__ __device__ inline bool
operator !=(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return !(a == b);
}

/************************************************************************
*  4-dimensional vectors or 4-by-C matrices				*
************************************************************************/
template <class T, size_t C> __host__ __device__ inline mat4x<T, C>
operator -(const mat4x<T, C>& a)
{
    return {-a.x, -a.y, -a.z, -a.w};
}

template <class T, size_t C> __host__ __device__ inline mat4x<T, C>&
operator +=(mat4x<T, C>& a, const mat4x<T, C>& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

template <class T, size_t C> __host__ __device__ inline mat4x<T, C>&
operator -=(mat4x<T, C>& a, const mat4x<T, C>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

template <class T, size_t C> __host__ __device__ inline mat4x<T, C>&
operator *=(mat4x<T, C>& a, T c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    a.w *= c;
    return a;
}

template <class T, size_t C> __host__ __device__ inline mat4x<T, C>
operator +(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template <class T, size_t C> __host__ __device__ inline mat4x<T, C>
operator -(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

template <class T, size_t C> __host__ __device__ inline mat4x<T, C>
operator *(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

template <class T, size_t C> __host__ __device__ inline mat4x<T, C>
operator /(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

template <class T, size_t C> __host__ __device__ inline mat4x<T, C>
operator *(const mat4x<T, C>& a, T c)
{
    return {a.x * c, a.y * c, a.z * c, a.w * c};
}

template <class T, size_t C> __host__ __device__ inline mat4x<T, C>
operator /(T c, const mat4x<T, C>& a)
{
    return {c / a.x, c / a.y, c / a.z, c / a.w};
}

template <class T, size_t C> __host__ __device__ inline bool
operator ==(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <class T, size_t C> __host__ __device__ inline bool
operator !=(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return !(a == b);
}

/************************************************************************
*  Multiplication and division by scalar				*
************************************************************************/
template <class VM> __host__ __device__ inline
std::enable_if_t<(size0<VM>() > 1), VM&>
operator /=(VM& a, element_t<VM> c)
{
    return a *= (element_t<VM>(1)/c);
}

template <class VM> __host__ __device__ inline
std::enable_if_t<(size0<VM>() > 1), VM>
operator *(element_t<VM> c, const VM& a)
{
    return a * c;
}

template <class VM> __host__ __device__ inline
std::enable_if_t<(size0<VM>() > 1), VM>
operator /(const VM& a, element_t<VM> c)
{
    return a * (element_t<VM>(1)/c);
}

/************************************************************************
*  Output functions							*
************************************************************************/
template <class T, size_t C> std::ostream&
operator <<(std::ostream& out, const mat2x<T, C>& a)
{
    return out << '[' << a.x << ' ' << a.y << ']';
}

template <class T, size_t C> std::ostream&
operator <<(std::ostream& out, const mat3x<T, C>& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ']';
}

template <class T, size_t C> std::ostream&
operator <<(std::ostream& out, const mat4x<T, C>& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ' ' << a.w << ']';
}

/************************************************************************
*  homogeneous()							*
************************************************************************/
template <class T> __host__ __device__ inline mat3x<T, 1>
homogeneous(const mat2x<T, 1>& a)
{
    return {a.x, a.y, T(1)};
}

template <class T> __host__ __device__ inline mat4x<T, 1>
homogeneous(const mat3x<T, 1>& a)
{
    return {a.x, a.y, a.z, T(1)};
}

/************************************************************************
*  inhomogeneous()							*
************************************************************************/
template <class T> __host__ __device__ inline mat2x<T, 1>
inhomogeneous(const mat3x<T, 1>& a)
{
    return {a.x / a.z, a.y / a.z};
}

template <class T> __host__ __device__ inline mat3x<T, 1>
inhomogeneous(const mat4x<T, 1>& a)
{
    return {a.x / a.w, a.y / a.w, a.z / a.w};
}

/************************************************************************
*  dot()								*
************************************************************************/
template <class T, size_t C> __host__ __device__ inline vec<T, C>
dot(const mat2x<T, 1>& a, const mat2x<T, C>& b)
{
    return a.x * b.x + a.y * b.y;
}

template <class T, size_t C> __host__ __device__ inline vec<T, C>
dot(const mat3x<T, 1>& a, const mat3x<T, C>& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <class T, size_t C> __host__ __device__ inline vec<T, C>
dot(const mat4x<T, 1>& a, const mat4x<T, C>& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <class T, size_t C, class VM> __host__ __device__ inline
std::enable_if_t<size0<VM>() == C, mat<T, 2, size1<VM>()> >
dot(const mat2x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a)};
}

template <class T, size_t C, class VM> __host__ __device__ inline
std::enable_if_t<size0<VM>() == C, mat<T, 3, size1<VM>()> >
dot(const mat3x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a), dot(m.z, a)};
}

template <class T, size_t C, class VM> __host__ __device__ inline
std::enable_if_t<size0<VM>() == C, mat<T, 4, size1<VM>()> >
dot(const mat4x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a), dot(m.z, a), dot(m.w, a)};
}

/************************************************************************
*  square()								*
************************************************************************/
template <class VM> __host__ __device__ inline
std::enable_if_t<size1<VM>() == 1, element_t<VM> >
square(const VM& a)
{
    return dot(a, a);
}

/************************************************************************
*  cross()								*
************************************************************************/
template <class T> __host__ __device__ inline mat3x<T, 1>
cross(const mat3x<T, 1>& a, const mat3x<T, 1>& b)
{
    return {a.y * b.z - a.z * b.y,
	    a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/************************************************************************
*  ext()								*
************************************************************************/
template <class T, size_t N> __host__ __device__ inline mat<T, 2, N>
ext(const vec<T, 2>& a, const vec<T, N>& b)
{
    return {a.x * b, a.y * b};
}

template <class T, size_t N> __host__ __device__ inline mat<T, 3, N>
ext(const vec<T, 3>& a, const vec<T, N>& b)
{
    return {a.x * b, a.y * b, a.z * b};
}

template <class T, size_t N> __host__ __device__ inline mat<T, 4, N>
ext(const vec<T, 4>& a, const vec<T, N>& b)
{
    return {a.x * b, a.y * b, a.z * b, a.w * b};
}

namespace device
{
/************************************************************************
*  atomic operations							*
************************************************************************/
template <class T, class OP> __device__ inline void
atomicOp(T& dst, const T& src, OP op)
{
    op(&dst, src);
}
    
template <class T, size_t C, class OP> __device__ inline void
atomicOp(mat2x<T, C>& dst, const mat2x<T, C>& src, OP op)
{
    atomicOp(dst.x, src.x, op);
    atomicOp(dst.y, src.y, op);
}
    
template <class T, size_t C, class OP> __device__ inline void
atomicOp(mat3x<T, C>& dst, const mat3x<T, C>& src, OP op)
{
    atomicOp(dst.x, src.x, op);
    atomicOp(dst.y, src.y, op);
    atomicOp(dst.z, src.z, op);
}
    
template <class T, size_t C, class OP> __device__ inline void
atomicOp(mat4x<T, C>& dst, const mat4x<T, C>& src, OP op)
{
    atomicOp(dst.x, src.x, op);
    atomicOp(dst.y, src.y, op);
    atomicOp(dst.z, src.z, op);
    atomicOp(dst.w, src.w, op);
}
    
}	// namespace device

/************************************************************************
*  class Projectivity<T, DO, DI>					*
************************************************************************/
template <class T, size_t DO, size_t DI>
class Projectivity : public mat<T, DO + 1, DI + 1>
{
  public:
    constexpr static size_t	DO1	= DO + 1;
    constexpr static size_t	DI1	= DI + 1;
    constexpr static size_t	NPARAMS = DO1 * DI1;
    constexpr static size_t	DOF	= NPARAMS - 1;

    using matrix_type	= mat<T, DO1, DI1>;
    using		typename matrix_type::element_type;
    using point_type	= vec<element_type, DO>;
    using ppoint_type	= vec<element_type, DO1>;

    constexpr static size_t	inDim()		{ return DI; }
    constexpr static size_t	outDim()	{ return DO; }
    constexpr static size_t	nparams()	{ return NPARAMS; }

    __host__ __device__
		Projectivity(const matrix_type& m)	:matrix_type(m)	{}

    __host__ __device__
    point_type	operator ()(const vec<T, DI>& p) const
		{
		    return (*this)(p);
		}
    __host__ __device__
    point_type	operator ()(const vec<T, DI1>& p) const
		{
		    return inhomogeneous(mapP(p));
		}
    __host__ __device__
    point_type	operator ()(T u, T v) const
    		{
    		    return inhomogeneous(mapP(u, v));
    		}

    __host__ __device__
    ppoint_type	mapP(const vec<T, DI>& p) const
		{
		    return mapP(homogeneous(p));
		}
    __host__ __device__
    ppoint_type	mapP(const vec<T, DI1>& p) const
		{
		    return dot(*this, p);
		}
    __host__ __device__
    ppoint_type	mapP(T u, T v) const
    		{
    		    return dot(*this, vec<T, 3>{u, v, T(1)});
    		}

    template <size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 2&& DI_ == 2, mat<T, 2, 4> >
		image_derivative0(T eH, T eV, T u, T v)
		{
		    return {{eH*u, eH*v, eH, eV*u},
			    {eV*v, eV, -(eH*u + eV*v)*u, -(eH*u + eV*v)*v}};
		}

    template <size_t DO_=DO, size_t DI_=DI> __host__
    std::enable_if_t<DO_ == 2&& DI_ == 2>
		compose(const TU::Array<T, DOF>& dt)
		{
		    auto	t0 = this->x.x;
		    auto	t1 = this->x.y;
		    auto	t2 = this->x.z;
		    this->x.x -= (t0*dt[0] + t1*dt[3] + t2*dt[6]);
		    this->x.y -= (t0*dt[1] + t1*dt[4] + t2*dt[7]);
		    this->x.z -= (t0*dt[2] + t1*dt[5]);

		    t0 = this->y.x;
		    t1 = this->y.y;
		    t2 = this->y.z;
		    this->y.x -= (t0*dt[0] + t1*dt[3] + t2*dt[6]);
		    this->y.y -= (t0*dt[1] + t1*dt[4] + t2*dt[7]);
		    this->y.z -= (t0*dt[2] + t1*dt[5]);

		    t0 = this->z.x;
		    t1 = this->z.y;
		    t2 = this->z.z;
		    this->z.x -= (t0*dt[0] + t1*dt[3] + t2*dt[6]);
		    this->z.y -= (t0*dt[1] + t1*dt[4] + t2*dt[7]);
		    this->z.z -= (t0*dt[2] + t1*dt[5]);
		}
};

/************************************************************************
*  class Affinity<T, DO, DI>						*
************************************************************************/
template <class T, size_t DO, size_t DI>
class Affinity : public mat<T, DO, DI + 1>
{
  public:
    constexpr static size_t	DO1	= DO + 1;
    constexpr static size_t	DI1	= DI + 1;
    constexpr static size_t	NPARAMS = DO * DI1;
    constexpr static size_t	DOF	= NPARAMS;

    using matrix_type	= mat<T, DO, DI1>;
    using		typename matrix_type::element_type;
    using point_type	= vec<element_type, DO>;
    using ppoint_type	= vec<element_type, DO1>;
    using param_type	= matrix_type;

    constexpr static size_t	inDim()		{ return DI; }
    constexpr static size_t	outDim()	{ return DO; }
    constexpr static size_t	nparams()	{ return NPARAMS; }

    __host__ __device__
		Affinity(const matrix_type& m)	:matrix_type(m)	{}

    __host__ __device__
    point_type	operator ()(const vec<T, DI>& p) const
		{
		    return (*this)(homogeneous(p));
		}
    __host__ __device__
    point_type	operator ()(const vec<T, DI1>& p) const
		{
		    return dot(*this, p);
		}
    __host__ __device__
    point_type	operator ()(T u, T v) const
    		{
    		    return (*this)(vec<T, 3>({u, v, T(1)}));
    		}

    __host__ __device__
    ppoint_type	mapP(const vec<T, DI>& p) const
		{
		    return homogeneous((*this)(p));
		}
    __host__ __device__
    ppoint_type	mapP(const vec<T, DI1>& p) const
		{
		    return homogeneous((*this)(p));
		}
    __host__ __device__
    ppoint_type	mapP(T u, T v) const
    		{
    		    return homogeneous((*this)(u, v));
    		}

    __host__ __device__
    void	update(const param_type& dt)
		{
		    *this -= dt;
		}

    template <size_t DO_=DO, size_t DI_=DI>
    __host__ __device__ static std::enable_if_t<DO_ == 2&& DI_ == 2, param_type>
		image_derivative0(T eH, T eV, T u, T v)
		{
		    return {{eH*u, eH*v}, {eV*u, eV*v}};
		}

    template <size_t DO_=DO, size_t DI_=DI> __host__
    std::enable_if_t<DO_ == 2&& DI_ == 2>
		compose(const TU::Array<T, DOF>& dt)
		{
		    auto	t0 = this->x.x;
		    auto	t1 = this->x.y;
		    this->x.x -= (t0*dt[0] + t1*dt[3]);
		    this->x.y -= (t0*dt[1] + t1*dt[4]);
		    this->x.z -= (t0*dt[2] + t1*dt[5]);

		    t0 = this->y.x;
		    t1 = this->y.y;
		    this->y.x -= (t0*dt[0] + t1*dt[3]);
		    this->y.y -= (t0*dt[1] + t1*dt[4]);
		    this->y.z -= (t0*dt[2] + t1*dt[5]);
		}
};

/************************************************************************
*  class Rigidity<T, D>							*
************************************************************************/
template <class T, size_t D>
class Rigidity : public Affinity<T, D, D>
{
  public:
    constexpr static size_t	NPARAMS = D*(D+1)/2;
    constexpr static size_t	DOF	= NPARAMS;

    using base_type	= Affinity<T, D, D>;
    using		typename base_type::matrix_type;

    constexpr static size_t	dim()		{ return D; }

  public:
    __host__ __device__
		Rigidity(const matrix_type& m)	:base_type(m)	{}

    template <size_t D_=D>
    __host__ __device__ static std::enable_if_t<D_ == 2, vec<T, 3> >
		image_derivative0(T eH, T eV, T u, T v)
		{
		    return {eH, eV, eV*u - eH*v};
		}

    template <size_t D_=D> __host__ std::enable_if_t<D_ == 2>
		compose(const TU::Array<T, DOF>& dt)
		{
		    const auto	Rt = rotation(dt[2]);

		    auto	r0 = this->x.x;
		    auto	r1 = this->x.y;
		    this->x.x  = r0*Rt[0][0] + r1*Rt[1][0];
		    this->x.y  = r0*Rt[0][1] + r1*Rt[1][1];
		    this->x.z -= (this->x.x*dt[0] + this->x.y*dt[1]);

		    r0 = this->y.x;
		    r1 = this->y.y;
		    this->y.x  = r0*Rt[0][0] + r1*Rt[1][0];
		    this->y.y  = r0*Rt[0][1] + r1*Rt[1][1];
		    this->y.z -= (this->y.x*dt[0] + this->y.y*dt[1]);
		}
};

/************************************************************************
*  TU::cuda::get_element_ptr()						*
************************************************************************/
template <class T> inline element_t<T>*
get_element_ptr(thrust::device_ptr<T> p)
{
    return reinterpret_cast<element_t<T>*>(p.get());
}

template <class T> inline const element_t<T>*
get_element_ptr(thrust::device_ptr<const T> p)
{
    return reinterpret_cast<const element_t<T>*>(p.get());
}

}	// namespace cuda

/************************************************************************
*  class to_vec<T>							*
************************************************************************/
//! カラー画素をCUDAベクトルへ変換する関数オブジェクト
/*!
  \param T	変換先のCUDAベクトルの型
*/
template <class T>
class to_vec
{
  public:
    template <class S_>
    T	operator ()(const S_& val) const
	{
	    return vec(val, std::integral_constant<size_t, size0<T>()>());
	}

  private:
    template <class S_>
    T	vec(const S_& val, std::integral_constant<size_t, 1>) const
	{
	    return T(val);
	}
    template <class S_>
    T	vec(const S_& val, std::integral_constant<size_t, 3>) const
	{
	    using elm_t	= element_t<T>;

	    return {elm_t(val), elm_t(val), elm_t(val)};
	}
    template <class E_>
    T	vec(const RGB_<E_>& rgb, std::integral_constant<size_t, 3>) const
	{
	    using elm_t	= element_t<T>;

	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b)};
	}
    template <class S_>
    T	vec(const S_& val, std::integral_constant<size_t, 4>) const
	{
	    using elm_t	= element_t<T>;

	    return {elm_t(val), elm_t(val), elm_t(val), elm_t(255)};
	}
    template <class E_>
    T	vec(const RGB_<E_>& rgb, std::integral_constant<size_t, 4>) const
	{
	    using elm_t	= element_t<T>;

	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b), elm_t(rgb.a)};
	}
};

/************************************************************************
*  struct from_vec<T>							*
************************************************************************/
template <class T>
struct from_vec
{
    template <class T_>
    T		operator ()(const T_& val) const
		{
		    return T(val);
		}
    template <class T_>
    T		operator ()(const cuda::mat2x<T_, 1>& yuv422) const
	    	{
		    return T(yuv422.y);
		}
    template <class T_>
    T		operator ()(const cuda::mat3x<T_, 1>& rgb) const
		{
		    return T(0.229f*rgb.x + 0.587f*rgb.y +0.114f*rgb.z);
		}
    template <class T_>
    T		operator ()(const cuda::mat4x<T_, 1>& rgba) const
		{
		    return T(0.229f*rgba.x + 0.587f*rgba.y +0.114f*rgba.z);
		}
};

template <class E>
struct from_vec<RGB_<E> >
{
    using elm_t	 = typename E::element_type;

    template <class T_>
    RGB_<E>	operator ()(const T_& val) const
		{
		    return {elm_t(val), elm_t(val), elm_t(val)};
		}
    template <class T_>
    RGB_<E>	operator ()(const cuda::mat3x<T_, 1>& rgb) const
	    	{
		    return {elm_t(rgb.x), elm_t(rgb.y), elm_t(rgb.z)};
		}
    template <class T_>
    RGB_<E>	operator ()(const cuda::mat4x<T_, 1>& rgba) const
		{
		    return {elm_t(rgba.x), elm_t(rgba.y),
			    elm_t(rgba.z), elm_t(rgba.w)};
		}
};

template <>
struct from_vec<YUV422>
{
    using elm_t	 = YUV422::element_type;

    template <class T_>
    YUV422	operator ()(const T_& val) const
		{
		    return {elm_t(val)};
		}
    template <class T_>
    YUV422	operator ()(const cuda::mat2x<T_, 1>& yuv422) const
		{
		    return {elm_t(yuv422.y), elm_t(yuv422.x)};
		}
};

}	// namespace TU
