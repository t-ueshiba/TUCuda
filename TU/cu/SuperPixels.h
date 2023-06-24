/*!
  \file		SuperPixels.h
  \brief	super pixelフィルタの定義と実装
*/
#pragma once
#include <cuda/std/cassert>
#include <TU/cu/BoxFilter.h>
#include <TU/cu/functional.h>
#include <TU/cu/chrono.h>
#include <TU/cu/Labeling.h>
#include <cub/block/block_reduce.cuh>

namespace TU
{
namespace cu
{
#if defined(__NVCC__)
namespace detail
{
/************************************************************************
*  class mergeable<T, L>						*
************************************************************************/
//! cellをマージしてregionにする際のマージ可能性を判定するクラス
template <class T, class L>
class mergeable
{
  public:
    using value_type	= T;
    using plane_type	= Plane<T>;
    using moment_type	= Moment<T>;
    using label_type	= L;

    template <class T_>
    using range2 = range<range_iterator<thrust::device_ptr<const T_> > >;
    using range1 = range<thrust::device_ptr<int> >;

  public:
		mergeable(range2<moment_type> moments,
			  range2<plane_type> planes,
			  value_type thresh_angle, value_type thresh_dist,
			  range1 nchanged)
		    :_estimator(), _moments(moments), _planes(planes),
		     _thresh_dot(cos(thresh_angle)),
		     _thresh_dist(thresh_dist * sin(thresh_angle)),
		     _nchanged(nchanged)
		{
		    if (_planes.size() != _moments.size() ||
			_planes.begin().size() != _moments.begin().size())
			printf("Mismatch!\n");
		}

    value_type	threshAngle() const
		{
		    return std::acos(_thresh_dot) / RAD;
		}
    value_type	setThreshAngle(value_type thresh_angle)
		{
		    const auto	thresh_dot  = _thresh_dot;
		    _thresh_dot  = std::cos(thresh_angle * RAD);
		    _thresh_dist *= std::sqrt((T(1) - _thresh_dot*_thresh_dot)/
					      (T(1) - thresh_dot*thresh_dot));
		    return threshAngle();
		}

    value_type	threshDist() const
		{
		    return _thresh_dist
			 / std::sqrt(T(1) - _thresh_dot*_thresh_dot);
		}
    value_type	setThreshDist(value_type thresh_dist)
		{
		    _thresh_dist = thresh_dist
				 * std::sqrt(T(1) - _thresh_dot*_thresh_dot);
		    return threshDist();
		}

    __device__
    bool	operator()(label_type label0, label_type label1) const
		{
		    if (label0 < 0 || label1 < 0)
			return false;

		    if (label0 == label1)
			return true;

		    const int	y0 = label0 / _moments.begin().size();
		    const int	x0 = label0 - y0 * _moments.begin().size();
		    const int	y1 = label1 / _moments.begin().size();
		    const int	x1 = label1 - y1 * _moments.begin().size();

		    assert(y0 < _moments.size() &&
			   x0 < _moments.begin().size() &&
			   y1 < _moments.size() &&
			   x1 < _moments.begin().size());

		    const auto	moment0 = _moments[y0][x0];
		    const auto	moment1 = _moments[y1][x1];
		    const auto	p_large = (moment0.npoints() >
					   moment1.npoints() ?
					   _planes[y0][x0] : _planes[y1][x1]);
		    const auto	p_small = (moment0.npoints() >
					   moment1.npoints() ?
					   _planes[y1][x1] : _planes[y0][x0]);
		    const auto	m_small = (moment0.npoints() >
					   moment1.npoints() ?
					   moment1 : moment0);

		    if ((dot(p_large.normal(),
			     p_small.normal()) > _thresh_dot) &&
			(abs(dot(p_large.normal(),
				 p_large.center() - p_small.center())) <
			 sqrt(m_small.npoints()) * _thresh_dist))
		    {
			::atomicAdd(&_nchanged[0], 1);
			return true;
		    }
		    else
		    	return false;
		}

  private:
    const plane_estimator<value_type>	_estimator;
    const range2<moment_type>		_moments;
    const range2<plane_type>		_planes;
    const value_type			_thresh_dot;
    const value_type			_thresh_dist;
    const range1			_nchanged;

    constexpr static value_type		RAD = M_PI/180.0;
};

}	// namespace detail

namespace device
{
/************************************************************************
*  __device__ functions							*
************************************************************************/
template <class T, size_t W> __host__ __device__ void
suppress_local_label(const T in[][W], T out[][W], int u, int v, int winSize)
{
    const int	winRadius = winSize/2;
    const auto	label  = in[v + winRadius][u + winRadius];
    T		diff_label;
    int		ndiff_labels = 0;

    for (int j = 0; j < winSize; ++j)
	for (int i = 0; i < winSize; ++i)
	{
	    const auto	tmp = in[v + j][u + i];

	    if (tmp != label)
	    {
		diff_label = tmp;
		++ndiff_labels;
	    }
	}

    if (ndiff_labels > (winSize - 1)*(winSize - 1))
	out[v + winRadius][u + winRadius] = diff_label;
}

/************************************************************************
*  __global__ functions							*
************************************************************************/
//! 与えられたpixel mapを正方形cellに分割し，その代表pixelを求める
/*!
  各cellをthreadに割り付けた2次元block毎の並列処理として実装される
  \param pixels [IN]	pixel mapが与えられる
  \param cells	[OUT]	pixel mapを正方形分割したcell内で最小のMSEを持つ
			pixelが代表点として返される
  \param cellSize	cellの一辺のサイズ(pixel数)
 */
template <class PIXEL, class CELL> __global__ void
initialize_cells(range<range_iterator<PIXEL> > pixels,		// IN
		 range<range_iterator<CELL> >  cells,		// OUT
		 int			       cellSize)
{
  // (x, y): cellの座標
    const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (y < cells.size() && x < cells.begin().size())
    {
      // (u0, v0): cell左上隅
	const int	u0 = __mul24(x, cellSize);
	const int	v0 = __mul24(y, cellSize);

      // cell中で最小MSEを持つpixelをこのcellの代表pixelとする
	auto		pixel_min = pixels[v0][u0];
	for (auto v = v0; v < v0 + cellSize; ++v)
	    for (auto u = u0; u < u0 + cellSize; ++u)
	    {
		const auto	pixel = pixels[v][u];

		if (pixel.mse() < pixel_min.mse())
		    pixel_min  = pixel;
	    }

	cells[y][x] = pixel_min;
    }
}

//! 各pixelを最も近い代表点を持つcellに所属させる
/*!
  各pixelをthreadに割り付けた2次元block毎の並列処理として実装される．
  各pixelを，自身が内包されるcellを中心とする3*3個の正方cellのうち，
  その代表点に最も近いものに所属させ，そのindexをラベルとして与える．
  feature値からbackgrouind pixelと判定される場合は，特別なラベル値-1を与える．
  \param features [IN]	feature mapが与えられる
  \param pixels	  [IN]	pixel mapが与えられる
  \param cells	  [IN]	cell mapが代表pixelのmapとして与えられる
  \param labels	  [OUT]	各pixelが所属するcellへのindexがlabelとして返される
  \param sqdist		2つのpixel間の自乗距離を求める演算子
  \param is_background	feature値がbackgroudであるか判定する演算子
  \param cellSize	cellの一辺のサイズ(pixel数)
 */
template <class SUPER_PIXELS, class FEATURE, class PIXEL, class LABEL,
	  class SQDIST_OP, class IS_BACKGROUND_OP>
__global__ void
associate_pixels_with_cells(range<range_iterator<FEATURE> > features,	// IN
			    range<range_iterator<PIXEL> >   pixels,	// IN
			    range<range_iterator<PIXEL> >   cells,	// IN
			    range<range_iterator<LABEL> >   labels,	// OUT
			    SQDIST_OP			    sqdist,
			    IS_BACKGROUND_OP		    is_background,
			    int				    cellSize)
{
    using pixel_type	= typename std::iterator_traits<PIXEL>::value_type;
    using element_type	= typename pixel_type::element_type;

  // (u0, v0): pixel blockの左上隅
    const int	u0 = __mul24(blockIdx.x, blockDim.x);
    const int	v0 = __mul24(blockIdx.y, blockDim.y);

  // (xs, ys): (u0, v0)を内包するcellの左上に位置するcell
    const int	ncellsV = cells.size();
    const int	ncellsH = cells.begin().size();
    const int	xs = ::max(u0/cellSize - 1, 0);
    const int	ys = ::max(v0/cellSize - 1, 0);

  // pixel block [u0, u0 + blocDim.x)*[v0, v0 + blockDim.y)を内包し．かつ
  // その周囲に1 cell分のマージンを加えたcell領域をshared memoryに転送
    __shared__ pixel_type	cells_s[1 + SUPER_PIXELS::CellDimYMax + 1]
				       [1 + SUPER_PIXELS::CellDimXMax + 1];
    loadTile(slice(cells.cbegin(),
		   ys, ::min((v0 + blockDim.y - 1)/cellSize + 2, ncellsV) - ys,
		   xs, ::min((u0 + blockDim.x - 1)/cellSize + 2, ncellsH) - xs),
	     cells_s);
    __syncthreads();

  // (u, v): 注目pixel
    const int	u = u0 + threadIdx.x;
    const int	v = v0 + threadIdx.y;

    if (v < pixels.size() && u < pixels.begin().size())
    {
	if (is_background(features[v][u]))	// (u, v)がbackgroundならば...
	    labels[v][u] = -1;			// 特別な値(-1)をlabelとする
	else
	{
	  // (x, y): (u, v)を内包するcell
	    const auto	x  = u/cellSize;
	    const auto	y  = v/cellSize;

	  // [x0, x1)*[y0, y1): (u, v)に最も近いcellの探索範囲
	    const auto	x0 = ::max(x - 1, 0);
	    const auto	y0 = ::max(y - 1, 0);
	    const auto	x1 = ::min(x + 2, ncellsH);
	    const auto	y1 = ::min(y + 2, ncellsV);

	  // (u, v)に最も近いcell (x_min, y_min)を求める
	    const auto	pixel   = pixels[v][u];
	    auto	sqd_min = maxval<element_type>;
	    int		x_min = x;
	    int		y_min = y;
	    for (auto yy = y0; yy < y1; ++yy)
		for (auto xx = x0; xx < x1; ++xx)
		{
		    const auto	sqd = sqdist(cells_s[yy - ys][xx - xs], pixel);
		    if (sqd < sqd_min)
		    {
			sqd_min = sqd;
			x_min   = xx;
			y_min   = yy;
		    }
		}

	  // (u, v)に最も近いcellのindexを(u, v)のlabelとする
	    labels[v][u] = (sqd_min != maxval<element_type> ?
			    y_min * ncellsH + x_min : -1);
	}
    }
}

//! 同一cellに所属するpixelのfeatureを加算する
/*!
  各cellに対してそれを中心とする3*3個のcellから成るwindowを設定し，windowを
  blockに分割する．各block内のpixelをthreadに割り付けた並列処理として実装する．

    - windowの中心cellを (blockIdx.x, blockIdx.y) で指定する．
    - window内でのblockを blockIdx.z で指定する．
    - block内でのpixelを (threadIdx.x, threadIdx.y) で指定する．

　cell_features3は，縦cell数 * 横cell数 * window内block数 の3次元mapである．
  associate_pixels_with_cells() によって中心cellに所属させられたpixelは
  必ずwindow内に存在する．各threadは，自分のpixelが block: blockIdx.z に
  所属し，かつ中心cell: (blockIdx.x, blockIdx.y) に所属していたら，
  そのfeature値を cell_features3[blockIdx.y][blockIdx.x][blockIdx.z] に
  加算する．
  \param features	[IN]  feature mapが与えられる
  \param labels		[IN]  pixelが所属するcellへのindexが与えられる
  \param cell_features3	[OUT] 加算されたfeatureが3次元mapとして返される
  \param cellSize	cellの一辺のサイズ(pixel数)
 */
template <class SUPER_PIXELS, class FEATURE, class LABEL, class CELL_FEATURE>
__global__ void
accumulate_cell_features(range<range_iterator<FEATURE> > features,
			 range<range_iterator<LABEL> >	 labels,
			 range<range_iterator<range_iterator<CELL_FEATURE> > >
							 cell_features3,
			 int				 cellSize)
{
    using feature_type	= typename std::iterator_traits<FEATURE>::value_type;
    using BlockReduce	= cub::BlockReduce<
				feature_type,
				SUPER_PIXELS::BlockDimX,
				cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
				SUPER_PIXELS::BlockDimY>;
    using TempStorage	= typename BlockReduce::TempStorage;

  // (x, y): window中心cellの座標
    const int	x = blockIdx.x;
    const int	y = blockIdx.y;

  // 3*3 cell windowの辺あたりのblock数
    const int	nblocksWinH = 3*cellSize / blockDim.x;

  // blockIdx.z: window内でのblockのindex
  // (bx, by):   window左上隅から見たblockの座標
    const int	by = blockIdx.z / nblocksWinH;
    const int	bx = blockIdx.z - by * nblocksWinH;

  // (du, dv): windowの左上隅から見た注目pixelの座標
    const int	du = __mul24(bx, blockDim.x) + threadIdx.x;
    const int	dv = __mul24(by, blockDim.y) + threadIdx.y;

  // 注目pixelが(x, y)に所属していたらそのfeature値を取り出す
    feature_type	feature(0);
    if (dv < 3*cellSize && du < 3*cellSize)
    {
      // (u, v): 画像原点から見た注目pixelの座標
	const int	u = __mul24(x - 1, cellSize) + du;
	const int	v = __mul24(y - 1, cellSize) + dv;

      // label: cell (x, y)のindex
	const int	label = y * cell_features3.begin().size() + x;

	if (0 <= v && v < features.size() &&
	    0 <= u && u < features.begin().size() &&
	    labels[v][u] == label)	// (u, v)が(x, y)に所属していたら
	    feature = features[v][u];	// feature値をバッファに登録
    }
    __syncthreads();

  // block中のpixelのうち中心cell (x, y)に所属するもののfeature値を総和
    __shared__ TempStorage	tmp;
    const auto	feature_sum = BlockReduce(tmp).Sum(feature);

    if (threadIdx.y == 0 && threadIdx.x == 0)
	cell_features3[y][x][blockIdx.z] = feature_sum;
}

//! block毎に総和された中心cellのfeature値を加算してcell単位の総和値にする
/*
  \param cell_features3	[IN]  block単位で総和されたfeatureのが3次元map
  \param cell_features	[OUT] cell単位で総和されたfeatureが返される
  \param cells		[OUT] 総和されたcell_featuresから計算されたpixelが
			      返される
 */
template <class FEATURE3, class FEATURE, class CELL, class CREATE_PIXEL_OP>
__global__ void
compute_cells(range<range_iterator<range_iterator<FEATURE3> > >
					      cell_features3,
	      range<range_iterator<FEATURE> > cell_features,
	      range<range_iterator<CELL> >    cells,
	      CREATE_PIXEL_OP		      create_pixel)
{
    using feature_type = typename std::iterator_traits<FEATURE>::value_type;

    const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (y < cells.size() && x < cells.begin().size())
    {
	const int	nblocksWin = (*cell_features3.begin()).begin().size();
	feature_type	cell_feature(0);
	for (int i = 0; i < nblocksWin; ++i)
	    cell_feature += cell_features3[y][x][i];

	cell_features[y][x] = cell_feature;
	cells[y][x]	    = create_pixel(cell_feature);
    }
}

template <class SUPER_PIXELS, class LABEL> __global__ void
suppress_local_cells(range<range_iterator<LABEL> > labels, int winSize)
{
    using label_type = typename std::iterator_traits<LABEL>::value_type;

    __shared__ label_type
	in_s[ SUPER_PIXELS::BlockDimY + SUPER_PIXELS::WinSizeMax]
	    [ SUPER_PIXELS::BlockDimX + SUPER_PIXELS::WinSizeMax],
	out_s[SUPER_PIXELS::BlockDimY + SUPER_PIXELS::WinSizeMax]
	     [SUPER_PIXELS::BlockDimX + SUPER_PIXELS::WinSizeMax];

    const int	u0 = __mul24(blockIdx.x, blockDim.x);
    const int	v0 = __mul24(blockIdx.y, blockDim.y);

    loadTile(slice(labels.cbegin(),
		   v0, ::min(blockDim.y + winSize, labels.size() - v0),
		   u0, ::min(blockDim.x + winSize, labels.begin().size() - u0)),
	     in_s);
    loadTile(make_range(&in_s[0][0], blockDim.y + winSize,
			SUPER_PIXELS::BlockDimX + SUPER_PIXELS::WinSizeMax,
			blockDim.x + winSize),
	     out_s);
    __syncthreads();

    const int	tu = threadIdx.x;
    const int	tv = threadIdx.y;
    const int	u  = u0 + tu;
    const int	v  = v0 + tv;

    if (v < labels.size() - winSize && u < labels.begin().size() - winSize)
    {
	suppress_local_label(in_s, out_s, tu, tv, winSize);
	__syncthreads();

	suppress_local_label(out_s, in_s, tu, tv, winSize);

	const int	winRadius = winSize/2;
	labels[v + winRadius][u + winRadius]
	    = in_s[tv + winRadius][tu + winRadius];
    }
}

//! 全cellに通し番号を振る
/*!
  全cellに対する2次元tile block毎の並列処理として実装される
  \param cell_labels	[OUT] 各cellに対して，自身のindexがlabelとして付される
 */
template <class LABEL> __global__ void
initialize_cell_labels(range<range_iterator<LABEL> > cell_labels)
{
    const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (y < cell_labels.size() && x < cell_labels.begin().size())
	cell_labels[y][x] = y * cell_labels.begin().size() + x;
}

//! 各pixelに付されたregion labelに従ってpixelとcellのlabelを更新する
/*!
  全pixelに対する2次元tile block毎の並列処理として実装される
  \param region_labels	[IN]	 各pixelについて，所属regionの代表pixelへの
				 indexを与える
  \param labels		[IN/OUT] 各pixelについて，所属cellへのindexを与えると
				 所属regionの代表cellへのindexに書き換えられ
				 て返される
  \param cell_labels	[OUT]	 各cellについて，所属regionの代表cellへの
				 indexが返される
 */
template <class REGION_LABEL, class LABEL, class CELL_LABEL> __global__ void
create_cell_labels(range<range_iterator<REGION_LABEL> >	region_labels,
		   range<range_iterator<LABEL> >	labels,
		   range<range_iterator<CELL_LABEL> >	cell_labels)
{
  // (u, v): pixelの座標
    const int	u = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	v = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

  // 全cell_labelをbackgroud label(-1)に初期化
    if (v < cell_labels.size() && u < cell_labels.begin().size())
	cell_labels[v][u] = -1;
    __syncthreads();

    if (v < region_labels.size() && u < region_labels.begin().size())
    {
      // (u, v)の所属regionの代表pixelへのindex
	const auto	region_label = region_labels[v][u];	// root pixel

	if (region_label >= 0)
	{
	  // (ur, vr): (u, v)の所属regionの代表pixelの座標
	    const int	vr = region_label / region_labels.begin().stride();
	    const int	ur = region_label - vr * region_labels.begin().stride();
	    assert(vr < labels.size() && ur < labels.begin().size());

	  // (u, v)の所属regionの代表pixelの所属cellのindex
	    const auto	new_label = labels[vr][ur];

	  // (u, v)の所属cellへのindex
	    const auto	old_label = labels[v][u];

	  // (u, v)のlabelを所属regionの代表cellのindexに置き換える
	    ::atomicExch(&labels[v][u], new_label);

	    if (old_label >= 0)
	    {
	      // (x, y): (u, v)に割り当てられていたcellの座標
		const int	y = old_label / cell_labels.begin().size();
		const int	x = old_label - y * cell_labels.begin().size();
		assert(y < cell_labels.size() &&
		       x < cell_labels.begin().size());

	      // (x, y)のlabelも(u, v)の所属regionの代表cellのindexに置き換える
		::atomicExch(&cell_labels[y][x], new_label);
	    }
	}
    }
}

//! regionへlabel付けされたcellからregionのfeature値とregionのpixel値を計算
/*
  全cellに対する2次元tile block毎の並列処理として実装される
  \param cell_labels	[IN]     各cellに対してその所属regionの代表cellへの
				 indexを与える
  \param cell_features	[IN/OUT] 同一regionに属するcellのfeature値の総和を返す
  \param cells		[OUT]    cell_featureの総和から計算したpixel値を返す
  \param create_pixel	featureからpixelを生成する演算子
 */
template <class LABEL, class FEATURE, class CELL, class CREATE_PIXEL_OP>
__global__ void
merge_cells(range<range_iterator<LABEL> >   cell_labels,	// IN
	    range<range_iterator<FEATURE> > cell_features,	// IN/OUT
	    range<range_iterator<CELL> >    cells,		// OUT
	    CREATE_PIXEL_OP		    create_pixel)
{
    using feature_type = typename std::iterator_traits<FEATURE>::value_type;

  // (x, y): cellの座標
    const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (y < cell_labels.size() && x < cell_labels.begin().size())
    {
      // 対象cell (x, y)が所属するregionの代表cellのindex
	const auto	cell_label = cell_labels[y][x];

	if (cell_label >= 0)
	{
	  // (xr, yr): cell (x, y)が所属するregionの代表cellの座標
	    const int	yr = cell_label / cell_labels.begin().size();
	    const int	xr = cell_label - yr * cell_labels.begin().size();

	    assert(yr < cell_features.size() &&
		   xr < cell_features.begin().size());

	    if (yr != y || xr != x)
	    {
	      // cell (x, y)のfeatureを代表cellのfeatureに加算
		atomicOp(&cell_features[yr][xr], cell_features[y][x],
			 std::plus<>());
		cell_features[y][x] = 0;
	    }
	}
	__syncthreads();

      // 総和されたregion feature値からregionを代表するpixel値を計算
	cells[y][x] = create_pixel(cell_features[y][x]);
    }
}

}	// namespace device
#endif

/************************************************************************
*  class SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP,		*
*		     BLOCK_TRAITS, CLOCK>				*
************************************************************************/
//! CUDAによる2次元boxフィルタを表すクラス
template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS=BlockTraits<16, 16>, class CLOCK=cu::clock>
class SuperPixels : public BLOCK_TRAITS, public Profiler<CLOCK>
{
  public:
    using create_feature_op	= CREATE_FEATURE_OP;
    using create_pixel_op	= CREATE_PIXEL_OP;
    using feature_type		= typename create_feature_op::result_type;
    using pixel_type		= typename create_pixel_op::result_type;
    using label_type		= int32_t;
    using link_type		= uint32_t;
    using label_iterator_type	= typename Array<label_type>::iterator;

    using filter_type	= TU::cu::BoxFilter2<
			      TU::cu::device::box_convolver<feature_type> >;
    using value_type	= typename feature_type::element_type;

    using		BLOCK_TRAITS::BlockDimX;
    using		BLOCK_TRAITS::BlockDimY;

    constexpr static auto	WinSizeMax  = filter_type::WinSizeMax;
    constexpr static size_t	CellSizeMin = 4;
    constexpr static size_t	CellDimXMax = BlockDimX/CellSizeMin;
    constexpr static size_t	CellDimYMax = BlockDimY/CellSizeMin;

    struct is_background_label
    {
        __device__ bool	operator ()(label_type label)	{ return label < 0; }
    };

  private:
    using profiler_t	= Profiler<CLOCK>;

    constexpr static value_type	RAD = M_PI/180.0;

  public:
		SuperPixels(size_t winSize, size_t cellSize,
			    size_t nIterations, size_t nMergingsMax)
		    :profiler_t(5),
		     _create_feature(), _create_pixel(),
		     _box_filter(winSize, winSize), _cellSize(cellSize),
		     _suppressWinSize(5),
		     _nIterations(nIterations), _nMergingsMax(nMergingsMax),
		     _thresh_angle(15.0 * RAD), _thresh_dist(0.0003),
		     _nchanged(1)
		{
		}

    size_t	winSize()	const	{ return _box_filter.winSizeV(); }
    size_t	setWinSize(size_t size)
		{
		    _box_filter.setWinSizeV(size).setWinSizeH(size);
		    return winSize();
		}

    size_t	cellSize()		const	{ return _cellSize; }
    size_t	setCellSize(size_t size)
		{
		    _cellSize = size;
		    return cellSize();
		}

    size_t	suppressWinSize()	const	{ return _suppressWinSize; }
    size_t	setSuppressWinSize(size_t size)
		{
		    _suppressWinSize = size;
		    return suppressWinSize();
		}

    size_t	nIterations()		const	{ return _nIterations; }
    size_t	setNIterations(size_t n)
		{
		    _nIterations = n;
		    return nIterations();
		}

    size_t	nMergingsMax()		const	{ return _nMergingsMax; }
    size_t	setNMergingsMax(size_t n)
		{
		    _nMergingsMax = n;
		    return nMergingsMax();
		}

    value_type	threshAngle() const
		{
		    return _thresh_angle / RAD;
		}
    value_type	setThreshAngle(value_type value)
		{
		    _thresh_angle = value * RAD;
		    return threshAngle();
		}

    value_type	threshDist() const
		{
		    return _thresh_dist;
		}
    value_type	setThreshDist(value_type value)
		{
		    _thresh_dist = value;
		    return threshDist();
		}

    template <class IN, class LABEL, class SQDIST_OP, class IS_BACKGROUND_OP>
    void	create(IN row, IN rowe, LABEL rowL,
		       SQDIST_OP sqdist, IS_BACKGROUND_OP is_background);

    const auto&	features()		const	{ return _features; }
    const auto&	pixels()		const	{ return _pixels; }
    const auto&	cells()			const	{ return _cells; }
    const auto&	cell_labels()		const	{ return _cell_labels; }
    const auto&	cell_features()		const	{ return _cell_features; }

  private:
    template <class IN>
    void	initialize_features_and_pixels(IN row, IN rowe)		;
    void	initialize_cells()					;
    template <class LABEL, class SQDIST_OP, class IS_BACKGROUND_OP>
    void	associate_pixels_with_cells(
		    LABEL rowL,
		    SQDIST_OP sqdist, IS_BACKGROUND_OP is_background)	;
    template <class LABEL>
    void	update_cells(LABEL rowL)				;
    template <class LABEL>
    void	suppress_local_cells(LABEL rowL)			;
    template <class LABEL>
    void	merge_cells(LABEL rowL)					;

  // For debugging
    template <class LABEL, class IS_BACKGROUND_OP>
    void	check_background(LABEL rowL,
				 IS_BACKGROUND_OP is_background) const	;
    template <class LABEL>
    TU::Array2<int>
		pixel_npoints(LABEL rowL)			 const	;
    TU::Array2<int>
		cell_npoints()					 const	;
    void	check_npoints(const TU::Array2<int>& npoints)	 const	;
    void	print_cell(int x, int y)			 const	;
    template <class LABEL>
    void	print_cell(LABEL rowL, int x, int y)		 const	;

  private:
    Labeling<BLOCK_TRAITS>	_labeling;

    const create_feature_op	_create_feature;
    const create_pixel_op	_create_pixel;
    filter_type			_box_filter;

    int				_cellSize;
    int				_suppressWinSize;
    size_t			_nIterations;
    size_t			_nMergingsMax;
    value_type			_thresh_angle;
    value_type			_thresh_dist;

    Array2<feature_type>	_features;
    Array2<pixel_type>		_pixels;

    Array2<pixel_type>		_cells;
    Array3<feature_type>	_cell_features3;
    Array2<feature_type>	_cell_features;
    Array2<label_type>		_cell_labels;

    Array2<label_type>		_labels;
    Array<int>			_nchanged;
};

#if defined(__NVCC__)
template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
template <class IN, class LABEL, class SQDIST_OP, class IS_BACKGROUND_OP> void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
create(IN row, IN rowe, LABEL rowL,
       SQDIST_OP sqdist, IS_BACKGROUND_OP is_background)
{
    profiler_t::start(0);
    initialize_features_and_pixels(row, rowe);

    profiler_t::start(1);
    initialize_cells();

    profiler_t::start(2);
    for (size_t n = 0; n < _nIterations; ++n)
    {
	associate_pixels_with_cells(rowL, sqdist, is_background);
    	update_cells(rowL);
    }

    profiler_t::start(3);
    suppress_local_cells(rowL);

    // check_background(rowL, is_background);
    // std::cerr << "----------------------" << std::endl;

    // check_npoints(pixel_npoints(rowL));
    // std::cerr << "======================" << std::endl;

    profiler_t::start(4);
    merge_cells(rowL);

    cudaDeviceSynchronize();
    profiler_t::nextFrame();
}

/*
 *  private member functions
 */
//! 入力2次元配列の各要素からfeatureとpixelを生成する
/*!
  \param row	入力2次元配列の先頭行へのiterator
  \param row	入力2次元配列の末尾の次の行へのiterator
 */
template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
template <class IN> void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
initialize_features_and_pixels(IN row, IN rowe)
{
    using	std::size;

  // Compute features for each pixel of the input image.
    _features.resize(std::distance(row, rowe), size(*row));
    transform2(row, rowe, _features.begin(), _create_feature);

  // Compute pixels from features.
    _pixels.resize(_features.nrow(), _features.ncol());
    fill2(_pixels.begin(), _pixels.end(), pixel_type::invalid_plane());
    _box_filter.convolve(_features.cbegin(), _features.cend(),
    			 make_range_iterator(make_assignment_iterator(
						 _create_pixel,
						 _pixels.begin()->begin()),
    					     _pixels.stride(), _pixels.ncol()),
    			 true);
}

//! 正方形cellの代表pixelを求める
template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK> void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
initialize_cells()
{
    _cells.resize(_pixels.nrow() / _cellSize,
		  _pixels.ncol() / _cellSize);
    const dim3	threads(BlockDimX, BlockDimY);
    const dim3	blocks(divUp(_cells.ncol(), threads.x),
		       divUp(_cells.nrow(), threads.y));
    device::initialize_cells<<<blocks, threads>>>(
    	cu::make_range(_pixels.cbegin(), _pixels.nrow()),
    	cu::make_range(_cells.begin(),   _cells.nrow()),
    	_cellSize);
    gpuCheckLastError();
}

//! 各pixelを自分に最も近い代表pixelを持つcellに所属させる
/*!
  \param rowL	[OUT] 各pixelが所属するcellを指すindexが返される
 */
template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
template <class LABEL, class SQDIST_OP, class IS_BACKGROUND_OP> void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
associate_pixels_with_cells(LABEL rowL,
			    SQDIST_OP sqdist, IS_BACKGROUND_OP is_background)
{
    const dim3	threads(BlockDimX, BlockDimY);
    const dim3	blocks(divUp(_pixels.ncol(), threads.x),
		       divUp(_pixels.nrow(), threads.y));
    device::associate_pixels_with_cells<SuperPixels><<<blocks, threads>>>(
	cu::make_range(_features.cbegin(), _features.nrow()),
	cu::make_range(_pixels.cbegin(),   _pixels.nrow()),
	cu::make_range(_cells.cbegin(),    _cells.nrow()),
	cu::make_range(rowL,		   _pixels.nrow()),
	sqdist, is_background, _cellSize);
    gpuCheckLastError();
}

//! 各cellについて自分に関連付けられたpixelから自身のfeatureと代表点を更新する
/*!
  \param rowL	[IN] 各pixelが所属するcellを指すindexを与える
 */
template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
template <class LABEL> void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
update_cells(LABEL rowL)
{
    const dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(_cells.ncol(), _cells.nrow(),
		       divUp(9*_cellSize*_cellSize, threads.x*threads.y));
    _cell_features3.resize(blocks.y, blocks.x, blocks.z);
    device::accumulate_cell_features<SuperPixels><<<blocks, threads>>>(
	cu::make_range(_features.cbegin(),	_features.nrow()),
	cu::make_range(rowL,			_features.nrow()),
	cu::make_range(_cell_features3.begin(), _cell_features3.size()),
	_cellSize);
    gpuCheckLastError();

    blocks.z = 1;
    _cell_features.resize(_cells.nrow(), _cells.ncol());
    device::compute_cells<<<blocks, threads>>>(
	cu::make_range(_cell_features3.cbegin(), _cell_features3.size()),
	cu::make_range(_cell_features.begin(),   _cell_features.nrow()),
	cu::make_range(_cells.begin(),		 _cells.nrow()),
	_create_pixel);
    gpuCheckLastError();
}

template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
template <class LABEL> void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
suppress_local_cells(LABEL rowL)
{
    if (_suppressWinSize < 2)
	return;

    const dim3	threads(BlockDimX, BlockDimY);
    const dim3	blocks(divUp(_pixels.ncol(), threads.x),
		       divUp(_pixels.nrow(), threads.y));
    device::suppress_local_cells<SuperPixels><<<blocks, threads>>>(
	cu::make_range(rowL, _pixels.nrow()), _suppressWinSize);
    gpuCheckLastError();
}

//! 類似したcellをマージして領域(region)を生成する．
/*!
  \param rowL	各pixelが所属するcellの代表pixelへのindexを与え，
		cellをマージして生成されたregionの代表cellへのindexを返す
*/
template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
template <class LABEL> void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
merge_cells(LABEL rowL)
{
    _cell_labels.resize(_cells.nrow(), _cells.ncol());

    if (_nMergingsMax == 0)
    {
      // cellのマージを行わない場合は，cellに通し番号を振る
	const dim3	threads(BlockDimX, BlockDimY);
	const dim3	blocks(divUp(_cell_labels.ncol(), threads.x),
			       divUp(_cell_labels.nrow(), threads.y));
	device::initialize_cell_labels<<<blocks, threads>>>(
	    cu::make_range(_cell_labels.begin(), _cell_labels.nrow()));
	gpuCheckLastError();

	return;
    }

    for (size_t n = 0; n < _nMergingsMax; ++n)
    {
      // 各pixelが所属するcellまたはregionの代表cellへのindexから成る
      // ラベル画像自体を，与えられたマージ基準に基づいてさらにラベリング
      // する．新たに得られるラベル画像は，新たなregionの代表(cellでは
      // なく)pixelへのindexから成る．
	_labels.resize(_pixels.nrow(), _pixels.ncol());
	_nchanged[0] = 0;
	_labeling.label(rowL, rowL + _labels.nrow(),
			_labels.begin(),
			detail::mergeable<value_type, label_type>(
			    cu::make_range(_cell_features.cbegin(),
					   _cell_features.nrow()),
			    cu::make_range(_cells.cbegin(), _cells.nrow()),
			    _thresh_angle, _thresh_dist,
			    cu::make_range(_nchanged.begin(),
					   _nchanged.size())),
			is_background_label());
	if (_nchanged[0] == 0)	// ラベリング前と結果が変化しなかったら脱出
	    break;

      // 各pixelが所属するregionの代表pixelへのindexから成るラベル画像から
      // 所属regionの代表cellへのindexから成るラベル画像を生成する．各cell
      // についてもそれが所属するregionの代表cellへのindexから成るラベル
      // 画像を生成する．
	const dim3	threads(BlockDimX, BlockDimY);
	dim3		blocks(divUp(_labels.ncol(), threads.x),
			       divUp(_labels.nrow(), threads.y));
	device::create_cell_labels<<<blocks, threads>>>(
	    cu::make_range(_labels.cbegin(),	 _labels.nrow()),
	    cu::make_range(rowL,		 _labels.nrow()),
	    cu::make_range(_cell_labels.begin(), _cell_labels.nrow()));
	gpuCheckLastError();

      // 各cellの所属regionの代表cellへのindexから成るラベル画像を用いて，
      // 同一regionに属するcellのfeature値の総和およびそれから計算された
      // regionのpixel値を求める．
	blocks.x = divUp(_cell_labels.ncol(), threads.x);
	blocks.y = divUp(_cell_labels.nrow(), threads.y);
	device::merge_cells<<<blocks, threads>>>(
	    cu::make_range(_cell_labels.cbegin(),  _cell_labels.nrow()),
	    cu::make_range(_cell_features.begin(), _cell_features.nrow()),
	    cu::make_range(_cells.begin(),	   _cells.nrow()),
	    _create_pixel);
	gpuCheckLastError();

	// check_npoints(npoints);
	// std::cerr << "==================" << std::endl;
    }
}

/*
 *  For debugging
 */
template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
template <class LABEL, class IS_BACKGROUND_OP> void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
check_background(LABEL rowL, IS_BACKGROUND_OP is_background) const
{
    for (int v = 0; v < _features.nrow(); ++v)
    {
    	bool	fail = false;

    	for (int u = 0; u < _features.ncol(); ++u)
    	{
	    const label_type	l = *((rowL + v)->begin() + u);
    	    const feature_type	f = _features[v][u];

    	    if (l >= 0)
    	    {
    		if (is_background(f))
    		{
    		    std::cerr << "label=" << l << '[' << f.w.z
    			      << "]@(" << u << ',' << v << ") ";
    		    fail = true;
    		}
    	    }
    	    else
    	    {
    		if (!is_background(f))
    		{
    		    std::cerr << "label=" << l << '[' << f.w.z
    			      << "]@(" << u << ',' << v << ") ";
    		    fail = true;
    		}
    	    }
    	}
    	if (fail)
    	    std::cerr << std::endl;
    }
}

template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
template <class LABEL> TU::Array2<int>
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
pixel_npoints(LABEL rowL) const
{
    TU::Array2<int>	npoints(_cells.nrow(), _cells.ncol());
    npoints = 0;
    for (int v = 0; v < _features.nrow(); ++v)
    	for (int u = 0; u < _features.ncol(); ++u)
    	{
    	    const label_type	l = *((rowL + v)->begin() + u);

    	    if (l >= 0)
    	    {
    		const int	y = l / _cells.ncol();
    		const int	x = l - y * _cells.ncol();
    		++npoints[y][x];
    	    }
    	}

    return npoints;
}

template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
TU::Array2<int>
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
cell_npoints() const
{
    TU::Array2<int>	npoints(_cell_labels.nrow(), _cell_labels.ncol());
    npoints = 0;
    for (int y = 0; y < _cell_labels.nrow(); ++y)
    {
	for (int x = 0; x < _cell_labels.ncol(); ++x)
	{
	    const label_type	l = _cell_labels[y][x];

	    if (l >= 0)
	    {
		const feature_type	f  = _cell_features[y][x];
		const int		yr = l / _cell_labels.ncol();
		const int		xr = l - yr * _cell_labels.ncol();
		std::cerr << f.w.z << ":(" << x << ',' << y << ")->["
			  << xr << ',' << yr << "] ";
		npoints[yr][xr] += int(f.w.z);
	    }
	}
	std::cerr << std::endl;
    }

    return npoints;
}

template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
check_npoints(const TU::Array2<int>& npoints) const
{
    for (int y = 0; y < _cell_features.nrow(); ++y)
    {
	bool	fail = false;
	for (int x = 0; x < _cell_features.ncol(); ++x)
	{
	    const feature_type	f = _cell_features[y][x];
	    if (npoints[y][x] != int(f.w.z))
	    {
		std::cerr << f.w.z << '[' << npoints[y][x]
			  << "]@(" << x << ',' << y << ") ";
		fail = true;
	    }
	}
	if (fail)
	    std::cerr << std::endl;
    }
}

template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
print_cell(int x, int y) const
{
    const pixel_type	cell = _cells[y][x];

    std::cerr << "\n=== cell(" << x << ',' << y << ") ===" << std::endl;
    std::cerr << cell << std::endl;
}

template <class CREATE_FEATURE_OP, class CREATE_PIXEL_OP,
	  class BLOCK_TRAITS, class CLOCK>
template <class LABEL> void
SuperPixels<CREATE_FEATURE_OP, CREATE_PIXEL_OP, BLOCK_TRAITS, CLOCK>::
print_cell(LABEL rowL, int x, int y) const
{
    const pixel_type	cell = _cells[y][x];

    std::cerr << "\n*** cell(" << x << ',' << y << ") ***" << std::endl;
    std::cerr << cell << std::endl;

    for (int v = 0; v < _features.nrow(); ++v)
    {
	bool	cr = false;

	for (int u = 0; u < _features.ncol(); ++u)
	{
    	    const label_type	l = *((rowL + v)->begin() + u);

	    if (l >= 0)
	    {
		const int	yc = l / _cells.ncol();
		const int	xc = l - yc * _cells.ncol();

		if (x == xc && y == yc)
		{
		    const feature_type	f = _features[v][u];

		    std::cerr << " [" << f.x.x << ',' << f.x.y << ',' << f.x.z
		    	      << "]@(" << u << ',' << v << ')' << std::endl;
		    // std::cerr << " [" << f.x.z
		    // 	      << "]@(" << u << ',' << v << ')' << std::endl;

		    cr = true;
		}
	    }
	}
	if (cr)
	    std::cerr << std::endl;
    }
}
#endif	// __NVCC__
}	// namespace cu
}	// namespace TU
