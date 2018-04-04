/*
 *  $Id: main.cu,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"
#include "TU/cuda/functional.h"
#include "TU/cuda/chrono.h"

//#define OP	cuda::det3x3
//#define OP	cuda::laplacian3x3
//#define OP	cuda::sobelAbs3x3
#define OP	cuda::maximal3x3
//#define OP	cuda::minimal3x3

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
	Image<in_t>	in;
	in.restore(cin);				// 原画像を読み込む
	in.save(cout);					// 原画像をセーブ

      // GPUによって計算する．
	cuda::Array2<in_t>	in_d(in);
	cuda::Array2<out_t>	out_d(in.nrow(), in.ncol());
	cuda::op3x3(in_d.cbegin(), in_d.cend(), out_d.begin(), OP<in_t>());
	cudaThreadSynchronize();

	Profiler<cuda::clock>	cuProfiler(1);
	constexpr size_t	NITER = 1000;
	for (size_t n = 0; n < NITER; ++n)		// フィルタリング
	{
	    cuProfiler.start(0);
	    cuda::op3x3(in_d.cbegin(), in_d.cend(), out_d.begin(), OP<in_t>());
	    cuProfiler.nextFrame();
	}
	cuProfiler.print(std::cerr);
	
	Image<out_t>	out(out_d);
	out.save(cout);					// 結果画像をセーブ

      // CPUによって計算する．
	Profiler<>	profiler(1);
	Image<out_t>	outGold;
	for (size_t n = 0; n < 10; ++n)
	{
	    outGold = in;
	    profiler.start(0);
	    op3x3(outGold.begin(), outGold.end(), OP<in_t>());
	    profiler.nextFrame();
	}
	profiler.print(cerr);
	outGold.save(cout);

      // 結果を比較する．
	const int	V = 160;
	for (size_t u = 0; u < out.width(); ++u)
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
