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

//#define OP_H	cuda::maximal3x3
//#define OP_D	thrust::greater
#define OP_H	cuda::minimal3x3
#define OP_D	thrust::less

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    using in_t	= float;
    using out_t	= float;
    
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
	in.save(cout);					// 原画像をセーブ

      // GPUによって計算する．
	cuda::ExtremaFilter2<in_t>	extrema(winSize, winSize);
	cuda::Array2<in_t>		in_d(in);
	cuda::Array2<out_t>		out_d(in_d.nrow(), in_d.ncol());
	cuda::Array2<int2>		pos_d(in_d.nrow(), in_d.ncol());

	extrema.convolve(in_d.cbegin(), in_d.cend(),
			 out_d.begin(), pos_d.begin(), OP_D<in_t>());
	cudaDeviceSynchronize();

	Profiler<cuda::clock>	cuProfiler(1);
	constexpr size_t	NITER = 1000;
	for (size_t n = 0; n < NITER; ++n)
	{
	    cuProfiler.start(0);
	    extrema.convolve(in_d.cbegin(), in_d.cend(),
			     out_d.begin(), pos_d.begin(), OP_D<in_t>());
	    cuProfiler.nextFrame();
	}
	cuProfiler.print(cerr);
	
	Image<out_t>	out(out_d);
	out.save(cout);					// 結果画像をセーブ
	std::cerr << "OK" << std::endl;

      // CPUによって計算する．
	Image<out_t>	outGold(in.nrow(), in.ncol());
	for (size_t v = 0; v < in.height() - winSize + 1; ++v)
	    for (size_t u = 0; u < in.width() - winSize + 1; ++u)
	    {
		in_t	minval = 100000;
		
		for (size_t vv = v ; vv < v + winSize; ++vv)
		    for (size_t uu = u ; uu < u + winSize; ++uu)
			if (in[vv][uu] < minval)
			    minval = in[vv][uu];
		outGold[v][u] = minval;
	    }
	outGold.save(cout);

      // 結果を比較する．
	const int	V = 160;
	for (u_int u = 1; u < out.width() - 1; ++u)
	    if (out[V][u] != outGold[V][u])
	    {
		cerr << ' ' << u << ":(" << out[V][u] << ',' << outGold[V][u]
		     << ')' << endl;
		cerr << slice<3, 3>(in, u-1, V-1);
	    }
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
