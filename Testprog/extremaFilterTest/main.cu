/*
 *  $Id: main.cu,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"
#include "TU/cuda/functional.h"
#include "TU/cuda/chrono.h"
#include "TU/cuda/ExtremaFilter.h"
#include <thrust/functional.h>

namespace TU
{
template <class T> void
doJob(const Image<T>& in, size_t winSize)
{
    using op_t	  = cuda::device::extreme_value<thrust::less<T> >;
    using value_t = typename op_t::result_type;
    
  // GPUによって計算する．
    const cuda::ExtremaFilter2<T>	extrema(winSize, winSize);
    const cuda::Array2<T>		in_d(in);
    cuda::Array2<T>			out_d(in_d.nrow(), in_d.ncol());
    cuda::Array2<int2>			pos_d(in_d.nrow(), in_d.ncol());

    extrema.convolve(in_d.cbegin(), in_d.cend(), out_d.begin(), op_t());
    cudaDeviceSynchronize();

    Profiler<cuda::clock>	cuProfiler(1);
    constexpr size_t		NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)
    {
	cuProfiler.start(0);
	extrema.convolve(in_d.cbegin(), in_d.cend(), out_d.begin(), op_t());
	cuProfiler.nextFrame();
    }
    cuProfiler.print(std::cerr);
	
    const Image<T>	out(out_d);
    out.save(std::cout);				// 結果画像をセーブ

  // CPUによって計算する．
    Image<T>	outGold(in.width(), in.height());
    for (size_t v = 0; v < in.height() - winSize + 1; ++v)
	for (size_t u = 0; u < in.width() - winSize + 1; ++u)
	{
	    T	minval = 1000;
		
	    for (size_t vv = v ; vv < v + winSize; ++vv)
		for (size_t uu = u ; uu < u + winSize; ++uu)
		    if (in[vv][uu] <= minval)
			minval = in[vv][uu];
	    outGold[v][u] = minval;
	}
    outGold.save(std::cout);

  // 結果を比較する．
    const int	V = 160;
    for (u_int u = 1; u < out.width() - 1; ++u)
	if (out[V][u] != outGold[V][u])
	{
	    std::cerr << ' ' << u << ":(" << out[V][u] << ',' << outGold[V][u]
		      << ')' << std::endl;
	    std::cerr << slice<3, 3>(in, u-1, V-1);
	}
}

template <class T> void
doJob2(const Image<T>& in)
{
    const cuda::ExtremaFilter2<T>	extrema(3, 3);
    const cuda::Array2<T>		in_d(in);
    cuda::Array2<T>			out_d(in_d.nrow(), in_d.ncol());
    cuda::Array2<int2>			pos_d(in_d.nrow(), in_d.ncol());

    extrema.extrema(in_d.cbegin(), in_d.cend(),
		    out_d.begin(), pos_d.begin(), thrust::greater<T>(), true);

    Array2<int2>	pos(pos_d);
    Image<T>		out(in.width(), in.height());
    out = 0;
    for (size_t v = 1; v < out.nrow() - 1; ++v)
    	for (size_t u = 1; u < out.ncol() - 1; ++u)
    	{
    	    const auto	p = pos[v][u];
    	    out[p.y][p.x] = in[p.y][p.x];
    	}
    out.save(std::cout);				// 結果画像をセーブ

    cuda::op3x3(in_d.cbegin(), in_d.cend(), out_d.begin(),
		cuda::maximal3x3<T>());
    Image<T>	out2(out_d);
    out2.save(std::cout);

  // 結果を比較する．
    std::cerr << std::setprecision(5);
    const int	V = 160;
    for (size_t u = 2; u < out.width() - 2; ++u)
	    if (out[V][u] != 0)
	    {
		std::cerr << '(' << u << ',' << V << "): " << out[V][u]
			  << std::endl;
		std::cerr << slice<5, 5>(in,  V-2, u-2);
		std::cerr << slice<5, 5>(pos, V-2, u-2);
		std::cerr << slice<5, 5>(out, V-2, u-2);
	    }

    // for (size_t v = 1; v < out.height() - 1; ++v)
    // 	for (size_t u = 1; u < out.width() - 1; ++u)
    // 	    if (v == V && out[v][u] != out2[v][u])
    // 	    {
    // 		std::cerr << '(' << u << ',' << v << "): "
    // 			  << out[v][u] << " != " << out2[v][u] << std::endl;
    // 		std::cerr << slice<3, 3>(in, v-1, u-1);
    // 	    }

    // const int	V = 160;
    // for (u_int u = 1; u < out.width() - 1; ++u)
    // 	if (out[V][u] != out2[V][u])
    // 	{
    // 	    std::cerr << ' ' << u << ":(" << out[V][u] << ',' << out2[V][u]
    // 		      << ')' << std::endl;
    // 	    std::cerr << slice<3, 3>(in, V-1, u-1);
    // 	}
}

template <class T> void
doJob3(const Image<T>& in, size_t winSize)
{
    using op_t	  = cuda::device::extreme_value_position<thrust::greater<T>,
							 int2>;
    using value_t = typename op_t::result_type;
    
    const cuda::ExtremaFilter2<value_t>	extrema(winSize, winSize);
    const cuda::Array2<T>		in_d(in);
    cuda::Array2<T>			out_d(in_d.nrow(), in_d.ncol());
    cuda::Array2<int2>			pos_d(in_d.nrow(), in_d.ncol());

    extrema.convolve(in_d.cbegin(), in_d.cend(),
		     cuda::make_range_iterator(
			 thrust::make_zip_iterator(out_d.begin(), pos_d.begin()),
			 thrust::make_tuple(out_d.stride(), pos_d.stride()),
			 out_d.size()),
		     op_t(), true);

    Image<T>		out(out_d);
    out.save(std::cout);				// 結果画像をセーブ

    Array2<int2>	pos(pos_d);
    Image<T>		out2(in.width(), in.height());
    for (size_t v = 0; v < out2.nrow(); ++v)
    {
    	for (size_t u = 0; u < out2.ncol(); ++u)
    	{
    	    const auto	p = pos[v][u];
    	    out2[v][u] = in[p.y][p.x];
    	}
    }
    out2.save(std::cout);

  // 結果を比較する．
    const int	V = 160;
    for (u_int u = 1; u < out.width() - 1; ++u)
	if (out[V][u] != out2[V][u])
	    std::cerr << ' ' << u << ":(" << out[V][u] << ',' << out2[V][u]
		      << ')' << std::endl;
}

}	// namespace TU

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    size_t		winSize = 3;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "w:")) != -1; )
	switch (c)
	{
	  case 'w':
	    winSize = atoi(optarg);
	    break;
	}
    
    try
    {
	using value_type = float;
    
	TU::Image<value_type>	in;
	in.restore(std::cin);
	in.save(std::cout);
	
	TU::doJob(in, winSize);
      //TU::doJob2(in);
      //TU::doJob3(in, winSize);
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
