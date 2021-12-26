/*
 *  $Id: main.cu,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"
#include "TU/cuda/functional.h"
#include "TU/cuda/chrono.h"
#include <thrust/functional.h>

//#define OP_H	cuda::maximal3x3
//#define OP_D	thrust::greater
#define OP_H	cuda::minimal3x3
#define OP_D	thrust::less

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
  //typedef u_char	out_t;
    typedef float	out_t;
    
    try
    {
	Image<in_t>	in;
	in.restore(cin);				// 原画像を読み込む
      //in.save(cout);					// 原画像をセーブ

      // GPUによって計算する．
	cuda::Array2<in_t>	in_d(in);
	cuda::Array2<out_t>	out_d(in_d.nrow(), in_d.ncol());

	cuda::suppressNonExtrema3x3(in_d.cbegin(), in_d.cend(),
				    out_d.begin(), OP_D<in_t>());
	cudaDeviceSynchronize();

	Profiler<cuda::clock>	cuProfiler(1);
	constexpr size_t	NITER = 1000;
	for (size_t n = 0; n < NITER; ++n)
	{
	    cuProfiler.start(0);
	    cuda::suppressNonExtrema3x3(in_d.cbegin(), in_d.cend(),
					out_d.begin(), OP_D<in_t>());
	    cuProfiler.nextFrame();
	}
	cuProfiler.print(cerr);

	Image<out_t>	out(out_d);
      //out.save(cout);					// 結果画像をセーブ

      // CPUによって計算する．
	Profiler<>	profiler(1);
	Image<out_t>	outGold;
	for (u_int n = 0; n < 10; ++n)
	{
	    outGold = in;
	    profiler.start(0);
	    op3x3(outGold.begin(), outGold.end(), OP_H<in_t>());
	    profiler.nextFrame();
	}
	profiler.print(cerr);
      //outGold.save(cout);

      // 結果を比較する．
	const int	V = 160;
	for (u_int u = 1; u < out.width() - 1; ++u)
	    if (out[V][u] != outGold[V][u])
	    {
		cerr << ' ' << u << ":(" << out[V][u] << ',' << outGold[V][u]
		     << ')' << endl;
		cerr << slice<3, 3>(in, V-1, u-1);
	    }
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
