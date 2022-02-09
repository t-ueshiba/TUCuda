/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <stdexcept>
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/algorithm.h"
#include "TU/cu/Array++.h"
#include "TU/cu/functional.h"
#include "TU/cu/algorithm.h"
#include "TU/cu/chrono.h"
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace TU
{
#if 0
template <class E> void
range_print(const E& expr)
{
    typedef typename is_range<E>::type	expr_is_range;
    
    for (const auto& x : expr)
	std::cout << ' ' << x;
    std::cout << std::endl;
}
#endif

template <class T>
//using allocator_t = cu::mapped_allocator<T>;
using allocator_t = thrust::system::cuda::experimental::pinned_allocator<T>;
    
template <class T>
class maximal3x3
{
  public:
    using result_type = T;

    maximal3x3(T nonMaximal=0)	:_nonMaximal(nonMaximal)	{}

    template <class ITER> result_type
    operator ()(ITER p, ITER c, ITER n) const
    {
	return ((c[1] > p[0]) && (c[1] > p[1]) && (c[1] > p[2]) &&
		(c[1] > c[0])		       && (c[1] > c[2]) &&
		(c[1] > n[0]) && (c[1] > n[1]) && (c[1] > n[2]) ?
		c[1] : _nonMaximal);
    }

  private:
    const T	_nonMaximal;
};
}

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;

  //using in_t	= u_char;
  //using out_t	= u_char;
    using in_t	= float;
    using out_t	= float;
    
    try
    {
	cudaSetDeviceFlags(cudaDeviceMapHost);

	Image<in_t, allocator_t<in_t> >	in;
	in.restore(cin);				// 原画像を読み込む
	in.save(cout);					// 原画像をセーブ

      // GPUによって計算する．
	Image<out_t, allocator_t<out_t> >	out(in.width(), in.height());
	cu::opNxM(in.cbegin(), in.cend(), out.begin(),
		  cu::maximal8<in_t>());
	cudaDeviceSynchronize();

	Profiler<cu::clock>	cuProfiler(1);
	constexpr size_t	NITER = 1000;
	for (size_t n = 0; n < NITER; ++n)		// フィルタリング
	{
	    cuProfiler.start(0);
	    cu::opNxM(in.cbegin(), in.cend(), out.begin(),
		      cu::maximal8<in_t>());
	    cuProfiler.nextFrame();
	}
	cuProfiler.print(cerr);
	
	out.save(cout);					// 結果画像をセーブ

      // CPUによって計算する．
	Profiler<>	profiler(1);
	Image<out_t>	outGold;
	for (u_int n = 0; n < 10; ++n)
	{
	    outGold = in;
	    profiler.start(0);
	    op3x3(outGold.begin(), outGold.end(), TU::maximal3x3<in_t>());
	    profiler.nextFrame();
	}
	profiler.print(cerr);
	outGold.save(cout);

      // 結果を比較する．
	const int	V = 160;
	for (u_int u = 0; u < out.width(); ++u)
	    cerr << ' ' << (out[V][u] - outGold[V][u]);
	cerr <<  endl;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
