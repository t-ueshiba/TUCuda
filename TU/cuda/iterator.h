/*!
  \file		iterator.h
  \brief	半精度浮動小数点に燗する各種アルゴリズムの定義と実装
*/
#pragma once

#include <type_traits>
#include <thrust/iterator/iterator_adaptor.h>
#include "TU/range.h"
#include "TU/cuda/tuple.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  TU::cuda::map_iterator<FUNC, ITER>					*
************************************************************************/
template <class FUNC, class ITER>
class map_iterator
    : public thrust::iterator_adaptor<
	map_iterator<FUNC, ITER>,
	ITER,
	std::decay_t<
	    decltype(cu_apply(std::declval<FUNC>(),
			      std::declval<typename std::iterator_traits<ITER>
						   ::reference>()))>,
	thrust::use_default,
	thrust::use_default,
	decltype(cu_apply(std::declval<FUNC>(),
			  std::declval<typename std::iterator_traits<ITER>
						   ::reference>()))>
{
  private:
    using ref	= decltype(cu_apply(std::declval<FUNC>(),
				    std::declval<
				        typename std::iterator_traits<ITER>
						    ::reference>()));
    using super	= thrust::iterator_adaptor<map_iterator,
					   ITER,
					   std::decay_t<ref>,
					   thrust::use_default,
					   thrust::use_default,
					   ref>;
    friend	class thrust::iterator_core_access;

  public:
    using	typename super::reference;

  public:
    __host__ __device__
		map_iterator(FUNC func, const ITER& iter)
		    :super(iter), _func(func)				{}

  private:
    __host__ __device__
    reference	dereference()	const	{ return cu_apply(_func, *super::base()); }

  private:
    FUNC	_func;	//!< 演算子
};

template <class FUNC, class ITER>
__host__ __device__ inline map_iterator<FUNC, ITER>
make_map_iterator(FUNC&& func, const ITER& iter)
{
    return {std::forward<FUNC>(func), iter};
}

template <class FUNC, class ITER0, class ITER1, class... ITERS>
__host__ __device__
inline map_iterator<
	FUNC, thrust::zip_iterator<thrust::tuple<ITER0, ITER1, ITERS...> > >
make_map_iterator(FUNC&& func,
		  const ITER0& iter0, const ITER1& iter1, const ITERS&... iters)
{
    return {std::forward<FUNC>(func),
	    thrust::make_zip_iterator(
		thrust::make_tuple(iter0, iter1, iters...))};
}

/************************************************************************
*  class TU::cuda::assignment_iterator<FUNC, ITER>			*
************************************************************************/
#if defined(__NVCC__)
namespace detail
{
  template <class FUNC, class ITER>
  class assignment_proxy
  {
    private:
      template <class T_>
      static auto	check_func(ITER iter, const T_& val, FUNC func)
			    -> decltype(func(*iter, val), std::true_type());
      template <class T_>
      static auto	check_func(ITER iter, const T_& val, FUNC func)
			    -> decltype(*iter = func(val), std::false_type());
      template <class T_>
      using is_binary_func	= decltype(check_func(std::declval<ITER>(),
						      std::declval<T_>(),
						      std::declval<FUNC>()));

    public:
      __host__ __device__
      assignment_proxy(const ITER& iter, const FUNC& func)
	  :_iter(iter), _func(func)					{}

      template <class T_> __host__ __device__
      std::enable_if_t<is_binary_func<T_>::value, assignment_proxy&>
			operator =(T_&& val)
			{
			    _func(*_iter, std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      std::enable_if_t<!is_binary_func<T_>::value, assignment_proxy&>
			operator =(T_&& val)
			{
			    *_iter  = _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator +=(T_&& val)
			{
			    *_iter += _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator -=(T_&& val)
			{
			    *_iter -= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator *=(T_&& val)
			{
			    *_iter *= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator /=(T_&& val)
			{
			    *_iter /= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator &=(T_&& val)
			{
			    *_iter &= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator |=(T_&& val)
			{
			    *_iter |= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator ^=(T_&& val)
			{
			    *_iter ^= _func(std::forward<T_>(val));
			    return *this;
			}

    private:
      const ITER&	_iter;
      const FUNC&	_func;
  };
}	// namespace detail

//! operator *()を左辺値として使うときに，この左辺値と右辺値に指定された関数を適用するための反復子
/*!
  \param FUNC	変換を行う関数オブジェクトの型
  \param ITER	変換結果の代入先を指す反復子
*/
template <class FUNC, class ITER>
class assignment_iterator
    : public thrust::iterator_adaptor<assignment_iterator<FUNC, ITER>,
				      ITER,
				      thrust::use_default,
				      thrust::use_default,
				      thrust::use_default,
				      detail::assignment_proxy<FUNC, ITER> >
{
  private:
    using super	= thrust::iterator_adaptor<
				assignment_iterator,
				ITER,
				thrust::use_default,
				thrust::use_default,
				thrust::use_default,
				detail::assignment_proxy<FUNC, ITER> >;
    friend	class thrust::iterator_core_access;

  public:
    using	typename super::reference;

  public:
    __host__ __device__
		assignment_iterator(const FUNC& func, const ITER& iter)
		    :super(iter), _func(func)				{}

    const auto&	functor()	const	{ return _func; }

  private:
    __host__ __device__
    reference	dereference()	const	{ return {super::base(), _func}; }

  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
#endif	// __NVCC__

template <class FUNC, class ITER>
__host__ __device__ inline assignment_iterator<FUNC, ITER>
make_assignment_iterator(FUNC&& func, const ITER& iter)
{
    return {std::forward<FUNC>(func), iter};
}

template <class FUNC, class ITER0, class ITER1, class... ITERS>
__host__ __device__
inline assignment_iterator<
	FUNC, thrust::zip_iterator<thrust::tuple<ITER0, ITER1, ITERS...> > >
make_assignment_iterator(FUNC&& func, const ITER0& iter0,
			 const ITER1& iter1, const ITERS&... iters)
{
    return {std::forward<FUNC>(func),
	    thrust::make_zip_iterator(
		thrust::make_tuple(iter0, iter1, iters...))};
}
}	// namespace cuda
}	// namespace TU

namespace thrust
{
/************************************************************************
*  thrust::stride(const ITER&)						*
************************************************************************/
template <class T> __host__ __device__ ptrdiff_t
stride(device_ptr<T>)							;

template <class HEAD, class TAIL> __host__ __device__ inline auto
stride(const detail::cons<HEAD, TAIL>& iter_tuple)
{
    return TU::cuda::tuple_transform([](const auto& iter)
				     { return stride(iter); }, iter_tuple);
}

template <class ITER_TUPLE> __host__ __device__ inline auto
stride(const zip_iterator<ITER_TUPLE>& iter)
    -> decltype(stride(iter.get_iterator_tuple()))
{
    return stride(iter.get_iterator_tuple());
}
}	// namespace thrust

namespace TU
{
namespace cuda
{
/************************************************************************
*  TU::cuda::stride(const ITER&)					*
************************************************************************/
template <class ITER0, class ITER1, class... ITERS>
__host__ __device__ inline auto
stride(const ITER0& iter0, const ITER1& iter1, const ITERS&... iters)
{
    return stride(thrust::make_tuple(iter0, iter1, iters...));
}

template <class FUNC, class ITER> __host__ __device__ inline auto
stride(const map_iterator<FUNC, ITER>& iter)
    -> decltype(stride(iter.base()))
{
    return stride(iter.base());
}

template <class FUNC, class ITER> __host__ __device__ inline auto
stride(const assignment_iterator<FUNC, ITER>& iter)
    -> decltype(stride(iter.base()))
{
    return stride(iter.base());
}

/************************************************************************
*  TU::cuda::advance_stride(ITER&, STRIDE)				*
************************************************************************/
template <class ITER> __host__ __device__ inline auto
advance_stride(ITER& iter, const iterator_stride<ITER>& stride)
    -> void_t<decltype(iter += stride)>
{
    iter += stride;
}

template <class ITER_TUPLE, class HEAD, class TAIL>
__host__ __device__ inline auto
advance_stride(thrust::zip_iterator<ITER_TUPLE>& iter,
	       const thrust::detail::cons<HEAD, TAIL>& stride)
{
    using tuple_t = std::decay_t<decltype(iter.get_iterator_tuple())>;

    tuple_for_each([](auto&& x, const auto& y){ advance_stride(x, y); },
		   const_cast<tuple_t&>(iter.get_iterator_tuple()), stride);
}

template <class ITER, class HEAD, class TAIL> __host__ __device__ inline auto
advance_stride(ITER& iter, const thrust::detail::cons<HEAD, TAIL>& stride)
    -> void_t<decltype(iter.base())>
{
    using base_t = std::decay_t<decltype(iter.base())>;

    advance_stride(const_cast<base_t&>(iter.base()), stride);
}

/************************************************************************
*  class TU::cuda::range<ITER>						*
************************************************************************/
template <class ITER>
class range
{
  public:
    using value_type	 = iterator_value<ITER>;
    using const_iterator = const_iterator_t<ITER>;
    
  public:
    __host__ __device__
		range(ITER begin, int size)
		    :_begin(begin), _size(size)			{}

		range()						= delete;
		range(const range&)				= default;
    __host__ __device__
    range&	operator =(const range& r)
		{
		  //assert(r.size() == size());
		    copy<0>(r._begin, _size, _begin);
		    return *this;
		}
		range(range&&)					= default;
    range&	operator =(range&&)				= default;
    
    __host__ __device__
    int		size()	  const	{ return _size; }
    __host__ __device__
    auto	begin()		{ return _begin; }
    __host__ __device__
    auto	end()		{ return _begin + _size; }
    __host__ __device__
    auto	begin()	  const	{ return const_iterator(_begin); }
    __host__ __device__
    auto	end()	  const	{ return const_iterator(_begin + _size); }
    __host__ __device__
    auto	cbegin()  const	{ return begin(); }
    __host__ __device__
    auto	cend()	  const	{ return end(); }
    __host__ __device__
    decltype(auto)
		operator [](int i) const
		{
		  //assert(i < size());
		    return *(_begin + i);
		}

  private:
    const ITER	_begin;
    const int	_size;
};

template <class T>
class range<thrust::device_ptr<T> >
{
  public:
    using value_type	 = iterator_value<thrust::device_ptr<T> >;
    using const_iterator = const_iterator_t<thrust::device_ptr<T> >;
    
  public:
    __host__ __device__
		range(thrust::device_ptr<T> p, int size)
		    :_begin(p.get()), _size(size)		{}

		range()						= delete;
		range(const range&)				= default;
    __host__ __device__
    range&	operator =(const range& r)
		{
		  //assert(r.size() == size());
		    copy<0>(r._begin, _size, _begin);
		    return *this;
		}
		range(range&&)					= default;
    range&	operator =(range&&)				= default;
    
    __host__ __device__
    int		size()	  const	{ return _size; }
    __host__ __device__
    auto	begin()		{ return _begin; }
    __host__ __device__
    auto	end()		{ return _begin + _size; }
    __host__ __device__
    auto	begin()	  const	{ return const_iterator(_begin); }
    __host__ __device__
    auto	end()	  const	{ return const_iterator(_begin + _size); }
    __host__ __device__
    auto	cbegin()  const	{ return begin(); }
    __host__ __device__
    auto	cend()	  const	{ return end(); }
    __host__ __device__
    decltype(auto)
		operator [](int i) const
		{
		  //assert(i < size());
		    return *(_begin + i);
		}

  private:
    T* const	_begin;
    int const	_size;
};

/************************************************************************
*  class TU::cuda::range_iterator<ITER>					*
************************************************************************/
template <class ITER>
class range_iterator
    : public thrust::iterator_adaptor<range_iterator<ITER>,
				      ITER,
				      range<ITER>,
				      thrust::use_default,
				      thrust::use_default,
				      range<ITER> >
{
  private:
    using super	= thrust::iterator_adaptor<range_iterator,
					   ITER,
					   range<ITER>,
					   thrust::use_default,
					   thrust::use_default,
					   range<ITER> >;
    
  public:
    using	typename super::reference;
    using	typename super::difference_type;
    using	stride_t = iterator_stride<ITER>;
    friend	class thrust::iterator_core_access;
	  
  public:
    __host__ __device__
		range_iterator(ITER iter, stride_t stride, int size)
		    :super(iter), _stride(stride), _size(size)
		{}
    __host__	range_iterator(const TU::range_iterator<ITER, 0, 0>& iter)
		    :range_iterator(iter->begin(), iter.stride(), iter.size())
		{}

    __host__ __device__
    int		size() const
		{
		    return _size;
		}
    __host__ __device__
    stride_t	stride() const
		{
		    return _stride;
		}
	
  private:
    __host__ __device__
    reference	dereference() const
		{
		    return {super::base(), size()};
		}
    __host__ __device__
    void	increment()
		{
		    advance_stride(super::base_reference(), stride());
		}
    __host__ __device__
    void	decrement()
		{
		    advance_stride(super::base_reference(), -stride());
		}
    __host__ __device__
    void	advance(difference_type n)
		{
		    advance_stride(super::base_reference(), n*stride());
		}
    __host__ __device__
    difference_type
		distance_to(const range_iterator& iter) const
		{
		    return (iter.base() - super::base()) / leftmost(stride());
		}
    __host__ __device__
    static auto	leftmost(ptrdiff_t stride) -> ptrdiff_t
		{
		    return stride;
		}
    template <class STRIDE_>
    __host__ __device__
    static auto	leftmost(const STRIDE_& stride)
		{
		    using	std::get;
		    
		    return leftmost(get<0>(stride));
		}

  private:
    const stride_t	_stride;
    const int		_size;
};
	
/************************************************************************
*  TU::cuda::make_range_iterator()					*
************************************************************************/
template <class ITER> __host__ __device__ inline range_iterator<ITER>
make_range_iterator(ITER iter, iterator_stride<ITER> stride, int size)
{
    return {iter, stride, size};
}
    
template <class ITER, class... SS> __host__ __device__ inline auto
make_range_iterator(ITER iter, iterator_stride<ITER> stride, int size, SS... ss)
{
    return make_range_iterator(make_range_iterator(iter, ss...), stride, size);
}

/************************************************************************
*  TU::cuda::make_range()						*
************************************************************************/
template <class ITER> __host__ __device__ inline range<ITER>
make_range(ITER iter, int size)
{
    return {iter, size};
}

template <class ITER, class... SS> __host__ __device__ inline auto
make_range(ITER iter, int size, SS... ss)
{
    return make_range(make_range_iterator(iter, ss...), size);
}

template <class ITER, class... SS> __host__ inline auto
make_range(const TU::range_iterator<ITER, 0, 0>& iter, int size, SS... ss)
{
    return make_range(range_iterator<ITER>(iter), size, ss...);
}

/************************************************************************
*  TU::cuda::slice()							*
************************************************************************/
namespace detail
{
  template <class ITER> __host__ __device__ inline ITER
  make_slice_iterator(ITER iter)
  {
      return iter;
  }

  template <class ITER, class... IS> __host__ __device__ inline auto
  make_slice_iterator(ITER iter, int idx, int size, IS... is)
  {
      return make_range_iterator(cuda::detail::make_slice_iterator(
				     (*iter).begin() + idx, is...),
				 iter.stride(), size);
  }
}	// namespace detail
    
template <class ITER, class... IS> __host__ __device__ inline auto
slice(ITER iter, int idx, int size, IS... is)
{
    return make_range(cuda::detail::make_slice_iterator(iter + idx, is...),
		      size);
}

template <class ITER, class... IS> __host__ inline auto
slice(const TU::range_iterator<ITER, 0, 0>& iter, int idx, int size, IS... is)
{
    return slice(range_iterator<ITER>(iter), idx, size, is...);
}

}	// namespace cuda
}	// namespace TU
