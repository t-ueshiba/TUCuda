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
    
  // GPUによって計算する．
    const cuda::ExtremaFilter2<T>	extrema(winSize, winSize);
    const cuda::Array2<T>		in_d(in);
    cuda::Array2<T>			out_d(in_d.nrow(), in_d.ncol());
    cuda::Array2<short2>		pos_d(in_d.nrow(), in_d.ncol());

    extrema.convolve(in_d.cbegin(), in_d.cend(),
		     out_d.begin(), thrust::less<T>());
    cudaDeviceSynchronize();

    Profiler<cuda::clock>	cuProfiler(1);
    constexpr size_t		NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)
    {
	cuProfiler.start(0);
	extrema.convolve(in_d.cbegin(), in_d.cend(),
			 out_d.begin(), thrust::less<T>());
	cuProfiler.nextFrame();
    }
    cuProfiler.print(std::cerr);
	
    const Image<T>	out(out_d);
    out.save(std::cout);				// 結果画像をセーブ

  // CPUによって計算する．
    Image<T>	outGold(in.nrow(), in.ncol());
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
    cuda::Array2<short2>		pos_d(in_d.nrow(), in_d.ncol());

    extrema.extrema(in_d.cbegin(), in_d.cend(),
		    out_d.begin(), pos_d.begin(), thrust::greater<T>(), false);

    Array2<short2>	pos(pos_d);
    Image<T>		out(in.width(), in.height());
    for (size_t v = 0; v < out.nrow(); ++v)
    {
      //std::cerr << "v=" << v << ':';
    	for (size_t u = 0; u < out.ncol(); ++u)
    	{
    	    const auto	p = pos[v][u];
    	  //std::cerr << p;
    	    out[v][u] = in[p.y][p.x];
    	}
      //std::cerr << std::endl;
    }
    out.save(std::cout);				// 結果画像をセーブ

    cuda::op3x3(in_d.cbegin(), in_d.cend(), out_d.begin(),
		cuda::maximal3x3<T>());
    out = out_d;
    out.save(std::cout);
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
	
      //TU::doJob(in, winSize);
	TU::doJob2(in);
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
