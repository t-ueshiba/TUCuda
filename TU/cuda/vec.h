/*!
  \file		vec.h
  \author	Toshio UESHIBA
  \brief	cudaベクトルクラスとその演算子の定義
*/
#ifndef TU_CUDA_VEC_H
#define TU_CUDA_VEC_H

#include <cstdint>
#include "TU/Image++.h"

namespace TU
{
namespace detail
{
  template <class VEC>	struct vec_traits;

  template <>		struct vec_traits<char2>
			{
			    using element_type =	int8_t;
			    constexpr static size_t	size = 2;
			};
  template <>		struct vec_traits<char3>
			{
			    using element_type =	int8_t;
			    constexpr static size_t	size = 3;
			};
  template <>		struct vec_traits<char4>
			{
			    using element_type =	int8_t;
			    constexpr static size_t	size = 4;
			};

  template <>		struct vec_traits<uchar2>
			{
			    using element_type =	uint8_t;
			    constexpr static size_t	size = 2;
			};
  template <>		struct vec_traits<uchar3>
			{
			    using element_type =	uint8_t;
			    constexpr static size_t	size = 3;
			};
  template <>		struct vec_traits<uchar4>
			{
			    using element_type =	uint8_t;
			    constexpr static size_t	size = 4;
			};

  template <>		struct vec_traits<short2>
			{
			    using element_type =	int16_t;
			    constexpr static size_t	size = 2;
			};
  template <>		struct vec_traits<short3>
			{
			    using element_type =	int16_t;
			    constexpr static size_t	size = 3;
			};
  template <>		struct vec_traits<short4>
			{
			    using element_type =	int16_t;
			    constexpr static size_t	size = 4;
			};

  template <>		struct vec_traits<ushort2>
			{
			    using element_type =	uint16_t;
			    constexpr static size_t	size = 2;
			};
  template <>		struct vec_traits<ushort3>
			{
			    using element_type =	uint16_t;
			    constexpr static size_t	size = 3;
			};
  template <>		struct vec_traits<ushort4>
			{
			    using element_type =	uint16_t;
			    constexpr static size_t	size = 4;
			};

  template <>		struct vec_traits<int2>
			{
			    using element_type =	int32_t;
			    constexpr static size_t	size = 2;
			};
  template <>		struct vec_traits<int3>
			{
			    using element_type =	int32_t;
			    constexpr static size_t	size = 3;
			};
  template <>		struct vec_traits<int4>
			{
			    using element_type =	int32_t;
			    constexpr static size_t	size = 4;
			};

  template <>		struct vec_traits<uint2>
			{
			    using element_type =	uint32_t;
			    constexpr static size_t	size = 2;
			};
  template <>		struct vec_traits<uint3>
			{
			    using element_type =	uint32_t;
			    constexpr static size_t	size = 3;
			};
  template <>		struct vec_traits<uint4>
			{
			    using element_type =	uint32_t;
			    constexpr static size_t	size = 4;
			};

  template <>		struct vec_traits<float2>
			{
			    using element_type =	float;
			    constexpr static size_t	size = 2;
			};
  template <>		struct vec_traits<float3>
			{
			    using element_type =	float;
			    constexpr static size_t	size = 3;
			};
  template <>		struct vec_traits<float4>
			{
			    using element_type =	float;
			    constexpr static size_t	size = 4;
			};

  template <class T, size_t N>	struct vec_t;

  template <>	struct vec_t<int8_t,   1>	{ using type = char1;	};
  template <>	struct vec_t<int8_t,   2>	{ using type = char2;	};
  template <>	struct vec_t<int8_t,   3>	{ using type = char3;	};
  template <>	struct vec_t<int8_t,   4>	{ using type = char4;	};
    
  template <>	struct vec_t<uint8_t,  1>	{ using type = uchar1;	};
  template <>	struct vec_t<uint8_t,  2>	{ using type = uchar2;	};
  template <>	struct vec_t<uint8_t,  3>	{ using type = uchar3;	};
  template <>	struct vec_t<uint8_t,  4>	{ using type = uchar4;	};

  template <>	struct vec_t<int16_t,  1>	{ using type = short1;	};
  template <>	struct vec_t<int16_t,  2>	{ using type = short2;	};
  template <>	struct vec_t<int16_t,  3>	{ using type = short3;	};
  template <>	struct vec_t<int16_t,  4>	{ using type = short4;	};

  template <>	struct vec_t<uint16_t, 1>	{ using type = ushort1;	};
  template <>	struct vec_t<uint16_t, 2>	{ using type = ushort2;	};
  template <>	struct vec_t<uint16_t, 3>	{ using type = ushort3;	};
  template <>	struct vec_t<uint16_t, 4>	{ using type = ushort4;	};

  template <>	struct vec_t<int32_t,  1>	{ using type = int1;	};
  template <>	struct vec_t<int32_t,  2>	{ using type = int2;	};
  template <>	struct vec_t<int32_t,  3>	{ using type = int3;	};
  template <>	struct vec_t<int32_t,  4>	{ using type = int4;	};

  template <>	struct vec_t<uint32_t, 1>	{ using type = uint1;	};
  template <>	struct vec_t<uint32_t, 2>	{ using type = uint2;	};
  template <>	struct vec_t<uint32_t, 3>	{ using type = uint3;	};
  template <>	struct vec_t<uint32_t, 4>	{ using type = uint4;	};

  template <>	struct vec_t<int64_t,  1>	{ using type = longlong1; };
  template <>	struct vec_t<int64_t,  2>	{ using type = longlong2; };

  template <>	struct vec_t<float,    1>	{ using type = float1;	};
  template <>	struct vec_t<float,    2>	{ using type = float2;	};
  template <>	struct vec_t<float,    3>	{ using type = float3;	};
  template <>	struct vec_t<float,    4>	{ using type = float4;	};

  template <>	struct vec_t<double,   1>	{ using type = double1;	};
  template <>	struct vec_t<double,   2>	{ using type = double2;	};
}	// namespace detail

namespace cuda
{
  template <class T, size_t N>
  using	vec = typename TU::detail::vec_t<T, N>::type;
}	// namespace cuda
    
}	// namespace TU

#if defined(__NVCC__)
/************************************************************************
*  2-dimensional vectors						*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC&>
operator +=(VEC& a, VEC b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC&>
operator -=(VEC& a, VEC b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}
    
template <class VEC, class T> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC&>
operator *=(VEC& a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    a.x *= c;
    a.y *= c;
    return a;
}
    
template <class VEC, class T> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC&>
operator /=(VEC& a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    a.x /= c;
    a.y /= c;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC>
operator +(VEC a, VEC b)
{
    return {a.x + b.x, a.y + b.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC>
operator -(VEC a, VEC b)
{
    return {a.x - b.x, a.y - b.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC>
operator *(VEC a, VEC b)
{
    return {a.x * b.x, a.y * b.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC>
operator /(VEC a, VEC b)
{
    return {a.x / b.x, a.y / b.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC>
operator *(VEC a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    return {a.x * c, a.y * c};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC>
operator *(typename TU::detail::vec_traits<VEC>::element_type c, VEC a)
{
    return {c * a.x, c * a.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, VEC>
operator /(VEC a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    return {a.x / c, a.y / c};
}
    
/************************************************************************
*  3-dimensional vectors						*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC&>
operator +=(VEC& a, VEC b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC&>
operator -=(VEC& a, VEC b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
    
template <class VEC, class T> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC&>
operator *=(VEC& a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}
    
template <class VEC, class T> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC&>
operator /=(VEC& a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC>
operator +(VEC a, VEC b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC>
operator -(VEC a, VEC b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC>
operator *(VEC a, VEC b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC>
operator /(VEC a, VEC b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC>
operator *(VEC a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    return {a.x * c, a.y * c, a.z * c};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC>
operator *(typename TU::detail::vec_traits<VEC>::element_type c, VEC a)
{
    return {c * a.x, c * a.y, c * a.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, VEC>
operator /(VEC a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    return {a.x / c, a.y / c, a.z / c};
}
    
/************************************************************************
*  4-dimensional vectors						*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC&>
operator +=(VEC& a, VEC b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC&>
operator -=(VEC& a, VEC b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}
    
template <class VEC, class T> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC&>
operator *=(VEC& a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    a.w *= c;
    return a;
}
    
template <class VEC, class T> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC&>
operator /=(VEC& a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
    a.w /= c;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC>
operator +(VEC a, VEC b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC>
operator -(VEC a, VEC b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC>
operator *(VEC a, VEC b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC>
operator /(VEC a, VEC b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC>
operator *(VEC a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    return {a.x * c, a.y * c, a.z * c, a.w * c};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC>
operator *(typename TU::detail::vec_traits<VEC>::element_type c, VEC a)
{
    return {c * a.x, c * a.y, c * a.z, c * a.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, VEC>
operator /(VEC a, typename TU::detail::vec_traits<VEC>::element_type c)
{
    return {a.x / c, a.y / c, a.z / c, a.w / c};
}
    
/************************************************************************
*  Output functions							*
************************************************************************/
template <class VEC>
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 2, std::ostream&>
operator <<(std::ostream& out, const VEC& a)
{
    return out << '[' << a.x << ' ' << a.y << ']';
}

template <class VEC>
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 3, std::ostream&>
operator <<(std::ostream& out, const VEC& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ']';
}

template <class VEC>
std::enable_if_t<TU::detail::vec_traits<VEC>::size == 4, std::ostream&>
operator <<(std::ostream& out, const VEC& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ' ' << a.w << ']';
}

namespace TU
{
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
	    using elm_t	= typename detail::vec_traits<VEC>::element_type;
	    
	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b)};
	}
    template <class E_>
    std::enable_if_t<E_::size == 4, VEC>
    	operator ()(const RGB_<E_>& rgb) const
	{
	    using elm_t	= typename detail::vec_traits<VEC>::element_type;
	    
	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b), elm_t(rgb.a)};
	}
};

/************************************************************************
*  struct vec_to_color<E>						*
************************************************************************/
//! CUDAベクトルをカラー画素へ変換する関数オブジェクト
/*!
  \param E	変換先のカラー画素の型
*/
template <class E>
struct vec_to_color
{
    template <class VEC_>
    std::enable_if_t<detail::vec_traits<VEC_>::size == 3, RGB_<E> >
	operator ()(const VEC_& v) const
	{
	    using elm_t = typename RGB_<E>::element_type;
	    
	    return {elm_t(v.x), elm_t(v.y), elm_t(v.z)};
	}
    template <class VEC_>
    std::enable_if_t<detail::vec_traits<VEC_>::size == 4, RGB_<E> >
    	operator ()(const VEC_& v) const
	{
	    using elm_t = typename RGB_<E>::element_type;
	    
	    return {elm_t(v.x), elm_t(v.y), elm_t(v.z), elm_t(v.w)};
	}
};

}	// namespace TU

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
	   TU::make_assignment_iterator(q, TU::vec_to_color<E>()));
#else
    TU::Array<VEC, N>	tmp(n);
    copy_n(p, (N ? N : n), tmp.begin());
    std::copy_n(tmp.cbegin(), (N ? N : n),
		TU::make_assignment_iterator(q, TU::vec_to_color<E>()));
#endif
}

}	// namespace thrust
#endif	// __NVCC__    
#endif	// !TU_CUDA_VEC_H
