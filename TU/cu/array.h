/*!
  Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
   * Neither the name of National Institute of Advanced Industrial
     Science and Technology (AIST) nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.

  Author: Toshio Ueshiba

  \file		array.h
  \brief	CUDAデバイス上の配列に関連するクラスの定義と実装
*/
#pragma once

#include <utility>

namespace TU
{
namespace cu
{
/************************************************************************
*  TU::cu::device::array<T, D>						*
************************************************************************/
template <class T, size_t D>
class array
{
  public:
    using value_type		= T;
    using reference		= T&;
    using const_reference	= const T&;
    using pointer		= T*;
    using const_pointer		= const T*;

    constexpr static size_t	D_SQEP = (D*(D+1))/2;
    
  private:
    template <size_t I=0, class DUMMY=void>
    struct for_each
    {
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(array& a, OP_&& op)
			{
			    op(a[I]);
			    for_each<I + 1>::apply(a, std::forward<OP_>(op));
			}
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(array& a, const array& b, OP_&& op)
			{
			    op(a[I], b[I]);
			    for_each<I + 1>::apply(a, b, std::forward<OP_>(op));
			}
	__host__ __device__ __forceinline__
	static T	dot(const array& a, const array& b)
			{
			    return a[I]*b[I] + for_each<I + 1>::dot(a, b);
			}
    };
    template <class DUMMY>
    struct for_each<D, DUMMY>
    {
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(array& a, OP_&& op)			{}
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(array& a, const array& b, OP_&& op)	{}
	__host__ __device__ __forceinline__
	static T	dot(const array& a, const array& b)	{ return T(0); }
    };

    template <size_t I=0, size_t J=0, class DUMMY=void>
    struct sqextpro
    {
	template <size_t D_> __host__ __device__ __forceinline__
	static void	apply(const array& in, array<T, D_>& out)
			{
			    out[D*I + J - (I*(I+1))/2] = in[I] * in[J];
			    sqextpro<I, J+1>::apply(in, out);
			}
	template <size_t D_, class OP_> __host__ __device__ __forceinline__
	static void	apply(const array& in, array<T, D_>& out, OP_&& op)
			{
			    out[D*I + J - (I*(I+1))/2] = op(in[I], in[J]);
			    sqextpro<I, J+1>::apply(in, out,
						    std::forward<OP_>(op));
			}
    };
    template <size_t I, class DUMMY>
    struct sqextpro<I, D, DUMMY>
    {
	template <size_t D_> __host__ __device__ __forceinline__
	static void	apply(const array& in, array<T, D_>& out)
			{
			    sqextpro<I+1, I+1>::apply(in, out);
			}
	template <size_t D_, class OP_> __host__ __device__ __forceinline__
	static void	apply(const array& in, array<T, D_>& out, OP_&& op)
			{
			    sqextpro<I+1, I+1>::apply(in, out,
						      std::forward<OP_>(op));
			}
    };
    template <class DUMMY>
    struct sqextpro<D, D, DUMMY>
    {
	template <size_t D_> __host__ __device__ __forceinline__
	static void	apply(const array& in, array<T, D_>& out)
			{
			}
	template <size_t D_, class OP_> __host__ __device__ __forceinline__
	static void	apply(const array& in, array<T, D_>& out, OP_&& op)
			{
			}
    };

    template <size_t D1, size_t I=0, size_t J=0>
    struct extpro
    {
	__host__ __device__ __forceinline__
	static void	apply(const array& in,
			      const array<T, D1>& in1, array<T, D*D1>& out)
			{
			    out[I*D1 + J] = in[I] * in1[J];
			    extpro<D1, I, J+1>::apply(in, in1, out);
			}
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(const array& in, const array<T, D1>& in1,
			      array<T, D*D1>& out, OP_&& op)
			{
			    out[I*D1 + J] = op(in[I], in1[J]);
			    extpro<D1, I, J+1>::apply(in, in1, out,
						      std::forward<OP_>(op));
			}
    };
    template <size_t D1, size_t I>
    struct extpro<D1, I, D1>
    {
	__host__ __device__ __forceinline__
	static void	apply(const array& in,
			      const array<T, D1>& in1, array<T, D*D1>& out)
			{
			    extpro<D1, I+1, 0>::apply(in, in1, out);
			}
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(const array& in, const array<T, D1>& in1,
			      array<T, D*D1>& out, OP_&& op)
			{
			    extpro<D1, I+1, 0>::apply(in, in1, out,
						      std::forward<OP_>(op));
			}
    };
    template <size_t D1>
    struct extpro<D1, D, D1>
    {
	__host__ __device__ __forceinline__
	static void	apply(const array& in,
			      const array<T, D1>& in1, array<T, D*D1>& out)
			{
			}
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(const array& in, const array<T, D1>& in1,
			      array<T, D*D1>& out, OP_&& op)
			{
			}
    };

  public:
    __host__ __device__ constexpr
    static int		size()				{ return D; }
    __host__ __device__
    const_reference	operator [](int i)	const	{ return _data[i]; }
    __host__ __device__
    reference		operator [](int i)		{ return _data[i]; }
    __host__ __device__
    const_pointer	data()			const	{ return _data; }
    __host__ __host__ __device__
    pointer		data()				{ return _data; }

    __host__ __device__
    array&		negate()
			{
			    for_each<>::apply(*this,
					      [] __host__ __device__
					      (T& x){ x = -x; });
			    return *this;
			}
    
    __host__ __device__
    array&		operator +=(const array& b)
			{
			    for_each<>::apply(*this, b,
					      [] __host__ __device__
					      (T& x, const T& y){ x += y; });
			    return *this;
			}
    __host__ __device__
    array&		operator -=(const array& b)
			{
			    for_each<>::apply(*this, b,
					      [] __host__ __device__
					      (T& x, const T& y){ x -= y; });
			    return *this;
			}
    __host__ __device__
    array&		operator *=(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (T& x){ x *= c; });
			    return *this;
			}
    __host__ __device__
    array&		operator /=(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (T& x){ x /= c; });
			    return *this;
			}
    __host__ __device__
    array&		operator %=(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (T& x){ x %= c; });
			    return *this;
			}
    __host__ __device__
    array&		operator &=(const array& b)
			{
			    for_each<>::apply(*this, b,
					      [] __host__ __device__
					      (T& x, const T& y){ x &= y; });
			    return *this;
			}
    __host__ __device__
    array&		operator &=(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (T& x){ x &= c; });
			    return *this;
			}
    __host__ __device__
    array&		operator |=(const array& b)
			{
			    for_each<>::apply(*this, b,
					      [] __host__ __device__
					      (T& x, const T& y){ x |= y; });
			    return *this;
			}
    __host__ __device__
    array&		operator |=(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (T& x){ x |= c; });
			    return *this;
			}
    __host__ __device__
    array&		operator ^=(const array& b)
			{
			    for_each<>::apply(*this, b,
					      [] __host__ __device__
					      (T& x, const T& y){ x ^= y; });
			    return *this;
			}
    __host__ __device__
    array&		operator ^=(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (T& x){ x ^= c; });
			    return *this;
			}
    __host__ __device__
    void		fill(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (T& x){ x = c; });
			}
    __host__ __device__
    value_type		dot(const array& b) const
			{
			    return for_each<>::dot(*this, b);
			}
    template <size_t D_=D_SQEP> __host__ __device__
    array<T, D_>	ext() const
			{
			    array<T, D_>	val;
			    sqextpro<>::apply(*this, val);
			    return val;
			}
    template <size_t D_=D_SQEP, class OP_> __host__ __device__
    array<T, D_>	ext(OP_&& op) const
			{
			    array<T, D_>	val;
			    sqextpro<>::apply(*this, val, std::forward<OP_>(op));
			    return val;
			}

    template <size_t D1_> __host__ __device__
    array<T, D*D1_>	ext(const array<T, D1_>& b) const
			{
			    array<T, D*D1_>	val;
			    extpro<D1_>::apply(*this, b, val);
			    return val;
			}
    template <size_t D1_, class OP_> __host__ __device__
    array<T, D*D1_>	ext(const array<T, D1_>& b, OP_&& op) const
			{
			    array<T, D*D1_>	val;
			    extpro<D1_>::apply(*this, b, val,
					       std::forward<OP_>(op));
			    return val;
			}

  public:	// Should be public as std::array<T, D>
    value_type	_data[D];
};

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator -(const array<T, D>& a)
{
    auto	val(a);
    return val.negate();
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator +(const array<T, D>& a, const array<T, D>& b)
{
    auto	val(a);
    return val += b;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator -(const array<T, D>& a, const array<T, D>& b)
{
    auto	val(a);
    return val -= b;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator *(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val *= c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator *(const T& c, const array<T, D>& a)
{
    return a*c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator /(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val /= c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator %(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val %= c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator &(const array<T, D>& a, const array<T, D>& b)
{
    auto	val(a);
    return val &= b;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator &(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val &= c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator &(const T& c, const array<T, D>& a)
{
    return a & c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator |(const array<T, D>& a, const array<T, D>& b)
{
    auto	val(a);
    return val |= b;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator |(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val |= c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator |(const T& c, const array<T, D>& a)
{
    return a | c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator ^(const array<T, D>& a, const array<T, D>& b)
{
    auto	val(a);
    return val ^= b;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator ^(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val ^= c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator ^(const T& c, const array<T, D>& a)
{
    return a ^ c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ T
dot(const array<T, D>& a, const array<T, D>& b)
{
    return a.dot(b);
}

template <class T, size_t D> std::ostream&
operator <<(std::ostream& out, const array<T, D>& a)
{
    for (size_t i = 0; i < a.size(); ++i)
	out << ' ' << a[i];
    return out;
}

}	// namespace cu
}	// namespace TU
