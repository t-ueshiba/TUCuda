/*!
  \file		vec.h
  \author	Toshio UESHIBA
  \brief	cudaベクトルクラスとその演算子の定義
*/
#ifndef TU_CUDA_VEC_H
#define TU_CUDA_VEC_H

#include <cstdint>
#include <thrust/device_ptr.h>
#include "TU/Image++.h"		// for TU::RGB_<E>

namespace TU
{
namespace cuda
{
namespace detail
{
  template <class T>
  constexpr std::integral_constant<size_t, 1>	size(T)			;
    
  constexpr std::integral_constant<size_t, 1>	size(char1)		;
  constexpr std::integral_constant<size_t, 2>	size(char2)		;
  constexpr std::integral_constant<size_t, 3>	size(char3)		;
  constexpr std::integral_constant<size_t, 4>	size(char4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(uchar1)		;
  constexpr std::integral_constant<size_t, 2>	size(uchar2)		;
  constexpr std::integral_constant<size_t, 3>	size(uchar3)		;
  constexpr std::integral_constant<size_t, 4>	size(uchar4)		;

  constexpr std::integral_constant<size_t, 1>	size(short1)		;
  constexpr std::integral_constant<size_t, 2>	size(short2)		;
  constexpr std::integral_constant<size_t, 3>	size(short3)		;
  constexpr std::integral_constant<size_t, 4>	size(short4)		;

  constexpr std::integral_constant<size_t, 1>	size(ushort1)		;
  constexpr std::integral_constant<size_t, 2>	size(ushort2)		;
  constexpr std::integral_constant<size_t, 3>	size(ushort3)		;
  constexpr std::integral_constant<size_t, 4>	size(ushort4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(int1)		;
  constexpr std::integral_constant<size_t, 2>	size(int2)		;
  constexpr std::integral_constant<size_t, 3>	size(int3)		;
  constexpr std::integral_constant<size_t, 4>	size(int4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(uint1)		;
  constexpr std::integral_constant<size_t, 2>	size(uint2)		;
  constexpr std::integral_constant<size_t, 3>	size(uint3)		;
  constexpr std::integral_constant<size_t, 4>	size(uint4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(float1)		;
  constexpr std::integral_constant<size_t, 2>	size(float2)		;
  constexpr std::integral_constant<size_t, 3>	size(float3)		;
  constexpr std::integral_constant<size_t, 4>	size(float4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(longlong1)		;
  constexpr std::integral_constant<size_t, 2>	size(longlong2)		;

  constexpr std::integral_constant<size_t, 1>	size(double1)		;
  constexpr std::integral_constant<size_t, 2>	size(double2)		;

  template <class T>
  constexpr std::integral_constant<size_t, 1>	ncol(T)			;
    
  constexpr int8_t				element_t(char1)	;
  constexpr int8_t				element_t(char2)	;
  constexpr int8_t				element_t(char3)	;
  constexpr int8_t				element_t(char4)	;

  constexpr uint8_t				element_t(uchar1)	;
  constexpr uint8_t				element_t(uchar2)	;
  constexpr uint8_t				element_t(uchar3)	;
  constexpr uint8_t				element_t(uchar4)	;

  constexpr int16_t				element_t(short1)	;
  constexpr int16_t				element_t(short2)	;
  constexpr int16_t				element_t(short3)	;
  constexpr int16_t				element_t(short4)	;

  constexpr uint16_t				element_t(ushort1)	;
  constexpr uint16_t				element_t(ushort2)	;
  constexpr uint16_t				element_t(ushort3)	;
  constexpr uint16_t				element_t(ushort4)	;

  constexpr int32_t				element_t(int1)		;
  constexpr int32_t				element_t(int2)		;
  constexpr int32_t				element_t(int3)		;
  constexpr int32_t				element_t(int4)		;

  constexpr uint32_t				element_t(uint1)	;
  constexpr uint32_t				element_t(uint2)	;
  constexpr uint32_t				element_t(uint3)	;
  constexpr uint32_t				element_t(uint4)	;

  constexpr float				element_t(float1)	;
  constexpr float				element_t(float2)	;
  constexpr float				element_t(float3)	;
  constexpr float				element_t(float4)	;
    
  constexpr int64_t				element_t(longlong1)	;
  constexpr int64_t				element_t(longlong2)	;

  constexpr double				element_t(double1)	;
  constexpr double				element_t(double2)	;

  template <class T, size_t N>	struct vec;

  template <>	struct vec<int8_t,   1>		{ using type = char1;	};
  template <>	struct vec<int8_t,   2>		{ using type = char2;	};
  template <>	struct vec<int8_t,   3>		{ using type = char3;	};
  template <>	struct vec<int8_t,   4>		{ using type = char4;	};
    
  template <>	struct vec<uint8_t,  1>		{ using type = uchar1;	};
  template <>	struct vec<uint8_t,  2>		{ using type = uchar2;	};
  template <>	struct vec<uint8_t,  3>		{ using type = uchar3;	};
  template <>	struct vec<uint8_t,  4>		{ using type = uchar4;	};

  template <>	struct vec<int16_t,  1>		{ using type = short1;	};
  template <>	struct vec<int16_t,  2>		{ using type = short2;	};
  template <>	struct vec<int16_t,  3>		{ using type = short3;	};
  template <>	struct vec<int16_t,  4>		{ using type = short4;	};

  template <>	struct vec<uint16_t, 1>		{ using type = ushort1;	};
  template <>	struct vec<uint16_t, 2>		{ using type = ushort2;	};
  template <>	struct vec<uint16_t, 3>		{ using type = ushort3;	};
  template <>	struct vec<uint16_t, 4>		{ using type = ushort4;	};

  template <>	struct vec<int32_t,  1>		{ using type = int1;	};
  template <>	struct vec<int32_t,  2>		{ using type = int2;	};
  template <>	struct vec<int32_t,  3>		{ using type = int3;	};
  template <>	struct vec<int32_t,  4>		{ using type = int4;	};

  template <>	struct vec<uint32_t, 1>		{ using type = uint1;	};
  template <>	struct vec<uint32_t, 2>		{ using type = uint2;	};
  template <>	struct vec<uint32_t, 3>		{ using type = uint3;	};
  template <>	struct vec<uint32_t, 4>		{ using type = uint4;	};

  template <>	struct vec<float,    1>		{ using type = float1;	};
  template <>	struct vec<float,    2>		{ using type = float2;	};
  template <>	struct vec<float,    3>		{ using type = float3;	};
  template <>	struct vec<float,    4>		{ using type = float4;	};

  template <>	struct vec<int64_t,  1>		{ using type = longlong1; };
  template <>	struct vec<int64_t,  2>		{ using type = longlong2; };

  template <>	struct vec<double,   1>		{ using type = double1;	};
  template <>	struct vec<double,   2>		{ using type = double2;	};
}	// namespace derail
    
template <class T, size_t N>
using vec = typename detail::vec<T, N>::type;

template <class T, size_t C>
struct mat2x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

    value_type	x, y;
};

template <class T, size_t C>
struct mat3x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

    value_type	x, y, z;
};

template <class T, size_t C>
struct mat4x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

    value_type	x, y, z, w;
};

namespace detail
{
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, 2>	size(mat2x<T, C>)	;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, 3>	size(mat3x<T, C>)	;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, 4>	size(mat4x<T, C>)	;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, C>	ncol(mat2x<T, C>)	;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, C>	ncol(mat3x<T, C>)	;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, C>	ncol(mat4x<T, C>)	;
  template <class MAT>
  constexpr typename MAT::element_type		element_t(MAT)		;
}	// namespace detail
    
template <class VM> constexpr static size_t
size()
{
    return decltype(detail::size(std::declval<VM>()))::value;
}

template <class VM> constexpr static size_t
ncol()
{
    return decltype(detail::ncol(std::declval<VM>()))::value;
}

template <class VM>
using element_t	= decltype(detail::element_t(std::declval<VM>()));
    
template <class T, size_t R, size_t C>
using mat = std::conditional_t<R == 2, mat2x<T, C>,
			       std::conditional_t<R == 3,
						  mat3x<T, C>, mat4x<T, C> > >;
}	// namespace cuda
}	// namespace TU

/*
 *  vec<T, N> は CUDA組み込みのベクトル型であり global namespace で定義
 *  されているためADLが効かないので，演算子を global namespace で定義
 *  する．
 */
/************************************************************************
*  2-dimensional vectors or 2-by-C matrices				*
************************************************************************/
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM&>
operator +=(VM& a, const VM& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM&>
operator -=(VM& a, const VM& b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM&>
operator *=(VM& a, TU::cuda::element_t<VM> c)
{
    a.x *= c;
    a.y *= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM&>
operator /=(VM& a, TU::cuda::element_t<VM> c)
{
    a.x /= c;
    a.y /= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM>
operator +(const VM& a, const VM& b)
{
    return {a.x + b.x, a.y + b.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM>
operator -(const VM& a, const VM& b)
{
    return {a.x - b.x, a.y - b.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM>
operator *(const VM& a, const VM& b)
{
    return {a.x * b.x, a.y * b.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM>
operator /(const VM& a, const VM& b)
{
    return {a.x / b.x, a.y / b.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM>
operator *(const VM& a, TU::cuda::element_t<VM> c)
{
    return {a.x * c, a.y * c};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM>
operator *(TU::cuda::element_t<VM> c, const VM& a)
{
    return {c * a.x, c * a.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 2, VM>
operator /(const VM& a, TU::cuda::element_t<VM> c)
{
    return {a.x / c, a.y / c};
}

/************************************************************************
*  3-dimensional vectors or 3-by-C matrices				*
************************************************************************/
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM&>
operator +=(VM& a, const VM& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM&>
operator -=(VM& a, const VM& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM&>
operator *=(VM& a, TU::cuda::element_t<VM> c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM&>
operator /=(VM& a, TU::cuda::element_t<VM> c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM>
operator +(const VM& a, const VM& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM>
operator -(const VM& a, const VM& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM>
operator *(const VM& a, const VM& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM>
operator /(const VM& a, const VM& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM>
operator *(const VM& a, TU::cuda::element_t<VM> c)
{
    return {a.x * c, a.y * c, a.z * c};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM>
operator *(TU::cuda::element_t<VM> c, const VM& a)
{
    return {c * a.x, c * a.y, c * a.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 3, VM>
operator /(const VM& a, TU::cuda::element_t<VM> c)
{
    return {a.x / c, a.y / c, a.z / c};
}

/************************************************************************
*  4-dimensional vectors or 4-by-C matrices				*
************************************************************************/
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM&>
operator +=(VM& a, const VM& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM&>
operator -=(VM& a, const VM& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM&>
operator *=(VM& a, TU::cuda::element_t<VM> c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    a.w *= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM&>
operator /=(VM& a, TU::cuda::element_t<VM> c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
    a.w /= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM>
operator +(const VM& a, const VM& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM>
operator -(const VM& a, const VM& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM>
operator *(const VM& a, const VM& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM>
operator /(const VM& a, const VM& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM>
operator *(const VM& a, TU::cuda::element_t<VM> c)
{
    return {a.x * c, a.y * c, a.z * c, a.w * c};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM>
operator *(TU::cuda::element_t<VM> c, const VM& a)
{
    return {c * a.x, c * a.y, c * a.z, c * a.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VM>() == 4, VM>
operator /(const VM& a, TU::cuda::element_t<VM> c)
{
    return {a.x / c, a.y / c, a.z / c, a.w / c};
}
    
/************************************************************************
*  Output functions							*
************************************************************************/
template <class VM>
std::enable_if_t<TU::cuda::size<VM>() == 2, std::ostream&>
operator <<(std::ostream& out, const VM& a)
{
    return out << '[' << a.x << ' ' << a.y << ']';
}

template <class VM>
std::enable_if_t<TU::cuda::size<VM>() == 3, std::ostream&>
operator <<(std::ostream& out, const VM& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ']';
}

template <class VM>
std::enable_if_t<TU::cuda::size<VM>() == 4, std::ostream&>
operator <<(std::ostream& out, const VM& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ' ' << a.w << ']';
}

namespace TU
{
namespace cuda
{
/************************************************************************
*  set_zero()								*
************************************************************************/
template <class T> __host__ __device__ inline
std::enable_if_t<std::is_arithmetic<T>::value>
set_zero(T& x)
{
    x = 0;
}

template <class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == 2>
set_zero(VM& a)
{
    set_zero(a.x);
    set_zero(a.y);
}

template <class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == 3>
set_zero(VM& a)
{
    set_zero(a.x);
    set_zero(a.y);
    set_zero(a.z);
}

template <class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == 4>
set_zero(VM& a)
{
    set_zero(a.x);
    set_zero(a.y);
    set_zero(a.z);
    set_zero(a.w);
}

/************************************************************************
*  homogeneous()							*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 2, vec<element_t<VEC>, 3> >
homogeneous(const VEC& a)
{
    return {a.x, a.y, element_t<VEC>(1)};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 3, vec<element_t<VEC>, 4> >
homogeneous(const VEC& a)
{
    return {a.x, a.y, a.z, element_t<VEC>(1)};
}
    
/************************************************************************
*  inhomogeneous()							*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 3, vec<element_t<VEC>, 2> >
inhomogeneous(const VEC& a)
{
    return {a.x / a.z, a.y / a.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 4, vec<element_t<VEC>, 3> >
inhomogeneous(const VEC& a)
{
    return {a.x / a.w, a.y / a.w, a.z / a.w};
}
    
/************************************************************************
*  dot()								*
************************************************************************/
template <class VEC, class VM, std::enable_if_t<ncol<VEC>() == 1 &&
						size<VEC>() == 2 &&
						size<VM>()  == 2>* = nullptr>
__host__ __device__ inline auto
dot(const VEC& a, const VM& b)
{
    return a.x * b.x + a.y * b.y;
}
    
template <class VEC, class VM, std::enable_if_t<ncol<VEC>() == 1 &&
						size<VEC>() == 3 &&
						size<VM>()  == 3>* = nullptr>
__host__ __device__ inline auto
dot(const VEC& a, const VM& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
    
template <class VEC, class VM, std::enable_if_t<ncol<VEC>() == 1 &&
						size<VEC>() == 4 &&
						size<VM>()  == 4>* = nullptr>
__host__ __device__ inline auto
dot(const VEC& a, const VM& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
    
template <class T, size_t C, class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == C,
		 std::conditional_t<ncol<VM>() == 1,
				    vec<T, 2>, mat<T, 2, ncol<VM>()> > >
dot(const mat2x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a)};
}
    
template <class T, size_t C, class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == C,
		 std::conditional_t<ncol<VM>() == 1,
				    vec<T, 3>, mat<T, 3, ncol<VM>()> > >
dot(const mat3x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a), dot(m.z, a)};
}

template <class T, size_t C, class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == C,
		 std::conditional_t<ncol<VM>() == 1,
				    vec<T, 4>, mat<T, 4, ncol<VM>()> > >
dot(const mat4x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a), dot(m.z, a), dot(m.w, a)};
}

/************************************************************************
*  cross()								*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 3, VEC>
cross(const VEC& a, const VEC& b)
{
    return {a.y * b.z - a.z * b.y,
	    a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
    
/************************************************************************
*  ext()								*
************************************************************************/
template <class VEC0, class VEC1> __host__ __device__ inline
std::enable_if_t<ncol<VEC0>() == 1 && size<VEC0>() == 2 && ncol<VEC1>() == 1,
		 mat<element_t<VEC0>, 2, size<VEC1>()> >
ext(const VEC0& a, const VEC1& b)
{
    return {a.x * b, a.y * b};
}
    
template <class VEC0, class VEC1> __host__ __device__ inline
std::enable_if_t<ncol<VEC0>() == 1 && size<VEC0>() == 3 && ncol<VEC1>() == 1,
		 mat<element_t<VEC0>, 3, size<VEC1>()> >
ext(const VEC0& a, const VEC1& b)
{
    return {a.x * b, a.y * b, a.z * b};
}
    
template <class VEC0, class VEC1> __host__ __device__ inline
std::enable_if_t<ncol<VEC0>() == 1 && size<VEC0>() == 4 && ncol<VEC1>() == 1,
		 mat<element_t<VEC0>, 4, size<VEC1>()> >
ext(const VEC0& a, const VEC1& b)
{
    return {a.x * b, a.y * b, a.z * b, a.w * b};
}
    
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
    template <class T_> __host__ __device__
    point_type	operator ()(T_ u, T_ v) const
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
    template <class T_> __host__ __device__
    ppoint_type	mapP(T_ u, T_ v) const
    		{
    		    return dot(*this, vec<T, 3>{T(u), T(v), T(1)});
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
    template <class T_> __host__ __device__
    point_type	operator ()(T_ u, T_ v) const
    		{
    		    return (*this)(vec<T, 3>({T(u), T(v), T(1)}));
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
    template <class T_> __host__ __device__
    ppoint_type	mapP(T_ u, T_ v) const
    		{
    		    return homogeneous((*this)(u, v));
    		}
    __host__ __device__
    void	update(const param_type& dt)
		{
		    *this -= dt;
		}
    template <size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 2&& DI_ == 2, param_type>
		image_derivative0(const vec<T, DI>& edge, const vec<T, DI1>& p)
		{
		    return ext(edge, p);
		}
    template <size_t DO_=DO, size_t DI_=DI> __host__ __device__
    std::enable_if_t<DO_ == 2&& DI_ == 2>
		compose(const param_type& dt)
		{
		    this->x -= (this->x.x*dt.x + this->x.y*dt.y);
		    this->y -= (this->y.x*dt.x + this->y.y*dt.y);
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
};
}	//  namespace cuda
    
/************************************************************************
*  struct color_to_vec<VEC>						*
************************************************************************/
//! カラー画素をCUDAベクトルへ変換する関数オブジェクト
/*!
  \param VEC	変換先のCUDAベクトルの型
*/
template <class VEC>
struct color_to_vec
{
    template <class E_>
    std::enable_if_t<E_::size == 3, VEC>
    	operator ()(const RGB_<E_>& rgb) const
	{
	    using elm_t	= cuda::element_t<VEC>;
	    
	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b)};
	}
    template <class E_>
    std::enable_if_t<E_::size == 4, VEC>
    	operator ()(const RGB_<E_>& rgb) const
	{
	    using elm_t	= cuda::element_t<VEC>;
	    
	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b), elm_t(rgb.a)};
	}
};

/************************************************************************
*  struct vec_to_color<COLOR>						*
************************************************************************/
//! CUDAベクトルをカラー画素へ変換する関数オブジェクト
/*!
  \param COLOR	変換先のカラー画素の型
*/
template <class COLOR>
struct vec_to_color
{
    template <class VEC_>
    std::enable_if_t<cuda::size<VEC_>() == 3, COLOR>
	operator ()(const VEC_& v) const
	{
	    using elm_t	= typename COLOR::element_type;
	    
	    return {elm_t(v.x), elm_t(v.y), elm_t(v.z)};
	}
    template <class VEC_>
    std::enable_if_t<cuda::size<VEC_>() == 4, COLOR>
    	operator ()(const VEC_& v) const
	{
	    using elm_t	= typename COLOR::element_type;
	    
	    return {elm_t(v.x), elm_t(v.y), elm_t(v.z), elm_t(v.w)};
	}
};

}	// namespace TU

#if defined(__NVCC__)
namespace thrust
{
/************************************************************************
*  algorithms overloaded for thrust::device_ptr<__half>			*
************************************************************************/
template <size_t N, class E, class VEC> inline void
copy(const TU::RGB_<E>* p, size_t n, device_ptr<VEC> q)
{
    copy_n(TU::make_map_iterator(TU::color_to_vec<VEC>(), p), (N ? N : n), q);
}

template <size_t N, class VEC, class E> inline void
copy(device_ptr<const VEC> p, size_t n, TU::RGB_<E>* q)
{
#if 0
    copy_n(p, (N ? N : n),
	   TU::make_assignment_iterator(q, TU::vec_to_color<TU::RGB_<E> >()));
#else
    TU::Array<VEC, N>	tmp(n);
    copy_n(p, (N ? N : n), tmp.begin());
    std::copy_n(tmp.cbegin(), (N ? N : n),
		TU::make_assignment_iterator(
		    TU::vec_to_color<TU::RGB_<E> >(), q));
#endif
}

}	// namespace thrust
#endif	// __NVCC__    
#endif	// !TU_CUDA_VEC_H
