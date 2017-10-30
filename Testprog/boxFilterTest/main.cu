/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/BoxFilter.h"
#include "TU/Profiler.h"
#include "TU/cuda/chrono.h"
#if 1
#  include "TU/cuda/BoxFilter.h"
#elif 1
#  include "TU/cuda/NewBoxFilter.h"
#else
#  include "TU/cuda/NeoBoxFilter.h"
#endif

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;

  //typedef u_char	in_t;
    typedef float	in_t;
  //typedef short	out_t;
    typedef float	out_t;
    
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
	Image<in_t>	in;
	in.restore(cin);				// 原画像を読み込む

      // GPUによって計算する．
	cuda::BoxFilter2<out_t, 15>	cudaFilter(winSize, winSize);
	cuda::Array2<in_t>		in_d(in);
	cuda::Array2<out_t>		out_d(in_d.nrow(), in_d.ncol());
	cudaFilter.convolve(in_d.cbegin(), in_d.cend(), out_d.begin());
	cudaThreadSynchronize();

	Profiler<cuda::clock>	cudaProfiler(1);
	constexpr size_t	NITER = 1000;
	for (size_t n = 0; n < NITER; ++n)
	{
	    cudaProfiler.start(0);
	    cudaFilter.convolve(in_d.cbegin(), in_d.cend(), out_d.begin());
	    cudaProfiler.nextFrame();
	}
	cudaProfiler.print(std::cerr);

	Image<out_t>	out(out_d);
	out *= 1.0/(winSize*winSize);
	out.save(cout);					// 結果画像をセーブ

      // CPUによって計算する．
	Profiler<>		profiler(1);
	Image<out_t>		outGold(in.width(), in.height());
	BoxFilter2<out_t>	filter(winSize, winSize);
	for (u_int n = 0; n < 10; ++n)
	{
	    profiler.start(0);
	    filter.convolve(in.cbegin(), in.cend(), outGold.begin());
	    profiler.nextFrame();
	}
	profiler.print(cerr);
	outGold *= 1.0/(winSize*winSize);
	outGold.save(cout);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
