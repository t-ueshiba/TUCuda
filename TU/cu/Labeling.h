/*!
  \file		Labeling.h
  \brief	labelingフィルタの定義と実装
*/
#pragma once

#include <TU/Profiler.h>
#include <TU/cu/Array++.h>
#include <TU/cu/algorithm.h>
#include <TU/cu/functional.h>
#include <TU/cu/chrono.h>

namespace TU
{
namespace cu
{
#if defined(__NVCC__)
namespace device
{
/************************************************************************
*  __device__ functions							*
************************************************************************/
template <class LABEL> __device__ __forceinline__ LABEL
root(const LABEL* labels, LABEL label)
{
    while (label > labels[label])
	label = labels[label];
    return label;
}

template <class LABEL> __device__ __forceinline__ void
merge(LABEL* labels, LABEL label0, LABEL label1, int& changed)
{
    const auto	root0 = root(labels, label0);
    const auto	root1 = root(labels, label1);

    if (root0 != root1)
    {
	const auto	root_min = ::min(root0, root1);
	const auto	root_max = ::max(root0, root1);

      // Replace label of root_max with root_min.
	atomicMin(&labels[root_max], root_min);

	changed = 1;
    }
}

/************************************************************************
*  __global__ functions							*
************************************************************************/
template<class LABELING, class LINK, class LABEL> __global__ void
label_tiles(range<range_iterator<LINK> >  links,
	    range<range_iterator<LABEL> > labels)
{
    using link_type	= typename std::iterator_traits<LINK>::value_type;
    using label_type	= typename std::iterator_traits<LABEL>::value_type;

    const int	tu = threadIdx.x;
    const int	tv = threadIdx.y;
    const int	u0 = blockIdx.x * blockDim.x;
    const int	v0 = blockIdx.y * blockDim.y;
    const int	u  = u0 + tu;
    const int	v  = v0 + tv;

    if (v >= links.size() || u >= links.begin().size())
	return;

    constexpr int		Stride_s = LABELING::BlockDimX + 1;
    __shared__ link_type	links_s[ LABELING::BlockDimY][Stride_s];
    __shared__ label_type	labels_s[LABELING::BlockDimY][Stride_s];

  // Load links and disable connections at right/bottom tile borders.
    auto	link = links[v][u];
    if (tu == blockDim.x - 1)
    	link &= ~LABELING::RIGHT;
    if (tv == blockDim.y - 1)
	link &= ~LABELING::LOWER;
    links_s[tv][tu] = link;

  // Initialize labels within a local tile block.
    labels_s[tv][tu] = tv * Stride_s + tu;
    __syncthreads();

  // Update local labels iteratively.
    for (;;)
    {
      // 1. Check connections to 4-neighbors.
	const auto	old_label = labels_s[tv][tu];
	auto		label	  = old_label;

	if (links_s[tv][tu] & LABELING::RIGHT)
	    label = ::min(label, labels_s[tv][tu+1]);
	if (links_s[tv][tu] & LABELING::LOWER)
	    label = ::min(label, labels_s[tv+1][tu]);
	if (0 < tu && links_s[tv][tu-1] & LABELING::RIGHT)
	    label = ::min(label, labels_s[tv][tu-1]);
	if (0 < tv && links_s[tv-1][tu] & LABELING::LOWER)
	    label = ::min(label, labels_s[tv-1][tu]);

      // 2. Is any value changed?
	int	changed = 0;
	if (label < old_label)
	{
	    labels_s[tv][tu] = label;
	    changed = 1;
	}

	if (!__syncthreads_or(changed))
	    break;

      // 3. Flatten equivalence trees.
	label = root(&labels_s[0][0], label);
	__syncthreads();

      // 4. Save flatten trees.
	labels_s[tv][tu] = label;
	__syncthreads();
    }

  // Convert labels from local tile space to global image space.
    const int	dv = labels_s[tv][tu] / Stride_s;
    const int	du = labels_s[tv][tu] - dv * Stride_s;
    labels[v][u] = (v0 + dv) * labels.begin().stride() + (u0 + du);
}

template <class LABELING, class LINK, class LABEL> __global__ void
merge_tiles(range<range_iterator<LINK> >  links,
	    range<range_iterator<LABEL> > labels,
	    int tileSize)
{
    const int	u0 = __mul24(blockIdx.x, 2*tileSize);
    const int	v0 = __mul24(blockIdx.y, 2*tileSize);
    const int	u1 = ::min(u0 + tileSize,   links.begin().size());
    const int	v1 = ::min(v0 + tileSize,   links.size());
    const int	u2 = ::min(u0 + 2*tileSize, links.begin().size());
    const int	v2 = ::min(v0 + 2*tileSize, links.size());

    const auto	lp = &labels[0][0];
    int		changed;
    do
    {
	changed = 0;

	if (u1 < u2)
	{
	  // Merge upper-left and upper-right tiles.
	    for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x)
		if (links[v][u1-1] & LABELING::RIGHT)
		    merge(lp, labels[v][u1-1], labels[v][u1], changed);
	}

	if (v1 < v2)
	{
	  // Merge upper-left and lower-left tiles.
	    for (int u = u0 + threadIdx.x; u < u1; u += blockDim.x)
		if (links[v1-1][u] & LABELING::LOWER)
		    merge(lp, labels[v1-1][u], labels[v1][u], changed);

	    if (u1 < u2)
	    {
	      // Merge upper-right and lower-right tiles.
		for (int u = u1 + threadIdx.x; u < u2; u += blockDim.x)
		    if (links[v1-1][u] & LABELING::LOWER)
			merge(lp, labels[v1-1][u], labels[v1][u], changed);

	      // Merge lower-left and lower-right tiles.
		for (int v = v1 + threadIdx.x; v < v2; v += blockDim.x)
		    if (links[v][u1-1] & LABELING::RIGHT)
			merge(lp, labels[v][u1-1], labels[v][u1], changed);
	    }
	}
    } while (__syncthreads_or(changed));
}

template <class LABEL> __global__ void
flatten_labels(range<range_iterator<LABEL> > labels)
{
    const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (y < labels.size() && x < labels.begin().size())
	labels[y][x] = root(&labels[0][0], labels[y][x]);
}

template <class IN, class LABEL, class IS_BACKGROUND> __global__ void
flatten_labels(range<range_iterator<IN> >    in,
	       range<range_iterator<LABEL> > labels,
	       IS_BACKGROUND is_background)
{
    const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (y < labels.size() && x < labels.begin().size())
	labels[y][x] = (is_background(in[y][x]) ? -1 :
			root(&labels[0][0], labels[y][x]));
}

}	// namespace device
#endif	// __NVCC__

/************************************************************************
*  class Labeling<BLOCK_TRAITS, CLOCK>					*
************************************************************************/
//! CUDAによる2次元boxフィルタを表すクラス
template <class BLOCK_TRAITS=BlockTraits<16, 16>, class CLOCK=cu::clock>
class Labeling : public BLOCK_TRAITS, public Profiler<CLOCK>
{
  private:
    using profiler_t	= Profiler<CLOCK>;
    using link_type	= uint32_t;

  public:
    template <class IS_LINKED>
    struct link_detector : public OperatorTraits<2, 2>
    {
	__host__ __device__
	link_detector(IS_LINKED is_linked) :_is_linked(is_linked)	{}

	template <class T_, size_t W_> __host__ __device__ link_type
	operator ()(int v, int u, int nrow, int ncol, T_ in[][W_]) const
	{
	    link_type	result = 0;
	    if (u + 1 < ncol)
		result |= _is_linked(in[v][u], in[v][u + 1]);
	    if (v + 1 < nrow)
		result |= (_is_linked(in[v][u], in[v + 1][u]) << 1);
	    return result;
	}

      private:
	const IS_LINKED	_is_linked;
    };

    using BLOCK_TRAITS::BlockDimX;
    using BLOCK_TRAITS::BlockDimY;

    constexpr static link_type	RIGHT = 0x01;
    constexpr static link_type	LOWER = 0x02;

  public:
		Labeling()	:profiler_t(4)				{}

    template <class IN, class OUT, class IS_LINKED,
	      class IS_BACKGROUND=std::nullptr_t>
    void	label(IN row, IN rowe, OUT rowL, IS_LINKED is_linked,
		      IS_BACKGROUND is_background=nullptr)		;

  private:
    template <class IN, class IS_LINKED>
    void	create_links(IN row, IN rowe, IS_LINKED is_linked)	;
    template <class LABEL>
    void	label_tiles(LABEL rowL)				const	;
    template <class LABEL>
    void	merge_tiles(LABEL rowL)				const	;
    template <class IN, class LABEL>
    void	flatten_labels(IN, LABEL rowL, std::nullptr_t)	const	;
    template <class IN, class LABEL, class IS_BACKGROUND>
    void	flatten_labels(IN row, LABEL rowL,
			       IS_BACKGROUND is_background)	const	;

  private:
    Array2<link_type>	_links;
};

#if defined(__NVCC__)
template<class BLOCK_TRAITS, class CLOCK>
template <class IN, class OUT, class IS_LINKED, class IS_BACKGROUND> void
Labeling<BLOCK_TRAITS, CLOCK>::label(IN row, IN rowe, OUT rowL,
				     IS_LINKED is_linked,
				     IS_BACKGROUND is_background)
{
  // Detect links between 4-neighboring pixels.
    profiler_t::start(0);
    create_links(row, rowe, is_linked);

  // Perform labeling for each tile.
    profiler_t::start(1);
    label_tiles(rowL);

  // Iteratively merge tiles.
    profiler_t::start(2);
    merge_tiles(rowL);

  // Flatten labels.
    profiler_t::start(3);
    flatten_labels(row, rowL, is_background);

    cudaDeviceSynchronize();
    profiler_t::nextFrame();
}

template<class BLOCK_TRAITS, class CLOCK>
template <class IN, class IS_LINKED> void
Labeling<BLOCK_TRAITS, CLOCK>::create_links(IN row, IN rowe,
					    IS_LINKED is_linked)
{
    using std::size;

    _links.resize(std::distance(row, rowe), size(*row));
    opNxM<BLOCK_TRAITS>(row, rowe, _links.begin(),
			link_detector<IS_LINKED>(is_linked));
}

template<class BLOCK_TRAITS, class CLOCK> template <class LABEL> void
Labeling<BLOCK_TRAITS, CLOCK>::label_tiles(LABEL rowL) const
{
    const dim3	threads(BlockDimX, BlockDimY);
    const dim3	blocks(divUp(_links.ncol(), threads.x),
		       divUp(_links.nrow(), threads.y));
    device::label_tiles<Labeling><<<blocks, threads>>>(
	cu::make_range(_links.cbegin(), _links.nrow()),
	cu::make_range(rowL,		_links.nrow()));
    gpuCheckLastError();
}

template<class BLOCK_TRAITS, class CLOCK> template <class LABEL> void
Labeling<BLOCK_TRAITS, CLOCK>::merge_tiles(LABEL rowL) const
{
    dim3	threads(BlockDimX, 1);
    dim3	blocks(divUp(_links.ncol(), BlockDimX),
		       divUp(_links.nrow(), BlockDimX));
    int		tileSize = BlockDimX;	// initial tile size in x-axis
    while (blocks.x > 1 || blocks.y > 1)
    {
	blocks.x = divUp(blocks.x, 2);
	blocks.y = divUp(blocks.y, 2);
	device::merge_tiles<Labeling><<<blocks, threads>>>(
	    cu::make_range(_links.cbegin(), _links.nrow()),
	    cu::make_range(rowL,	    _links.nrow()),
	    tileSize);
	gpuCheckLastError();

	threads.x = std::min(threads.x << 1, uint32_t(BlockDimX * BlockDimX));
	tileSize <<= 1;
    }
}

template<class BLOCK_TRAITS, class CLOCK>
template <class IN, class LABEL> void
Labeling<BLOCK_TRAITS, CLOCK>::flatten_labels(IN, LABEL rowL,
					      std::nullptr_t) const
{
    const dim3	threads(BlockDimX, BlockDimY);
    const dim3	blocks(divUp(_links.ncol(), threads.x),
		       divUp(_links.ncol(), threads.y));
    device::flatten_labels<<<blocks, threads>>>(
	cu::make_range(rowL, _links.nrow()));
    gpuCheckLastError();
}

template<class BLOCK_TRAITS, class CLOCK>
template <class IN, class LABEL, class IS_BACKGROUND> void
Labeling<BLOCK_TRAITS, CLOCK>::flatten_labels(IN row, LABEL rowL,
					      IS_BACKGROUND is_background) const
{
    const dim3	threads(BlockDimX, BlockDimY);
    const dim3	blocks(divUp(_links.ncol(), threads.x),
		       divUp(_links.ncol(), threads.y));
    device::flatten_labels<<<blocks, threads>>>(
	cu::make_range(row , _links.nrow()),
	cu::make_range(rowL, _links.nrow()),
	is_background);
    gpuCheckLastError();
}

#endif	// __NVCC__
}	// namespace cu
}	// namespace TU
