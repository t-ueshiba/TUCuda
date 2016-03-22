/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <stdexcept>
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/algorithm.h"
#include "TU/cuda/Array++.h"
#include "TU/cuda/functional.h"
#include "TU/cuda/algorithm.h"
#include <cuda_runtime.h>
#include <cutil.h>

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

  //typedef u_char	in_t;
    typedef float	in_t;
  //typedef u_char	out_t;
    typedef float	out_t;
    
    try
    {
	Image<in_t>	in;
	in.restore(cin);				// 原画像を読み込む
	in.save(cout);					// 原画像をセーブ

      // GPUによって計算する．
	cuda::Array2<in_t>	in_d(in);
	cuda::Array2<out_t>	out_d(in.nrow(), in.ncol());

	u_int		timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timer));		// タイマーを作成
	cuda::op3x3(in_d.cbegin(), in_d.cend(), out_d.begin(), OP<in_t>());
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUT_SAFE_CALL(cutStartTimer(timer));
	u_int	NITER = 1000;
	for (u_int n = 0; n < NITER; ++n)		// フィルタリング
	    cuda::op3x3(in_d.cbegin(), in_d.cend(), out_d.begin(), OP<in_t>());
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(timer));

	cerr << float(NITER * 1000) / cutGetTimerValue(timer) << "fps" << endl;
	CUT_SAFE_CALL(cutDeleteTimer(timer));		// タイマーを消去

	Image<out_t>	out;
	out_d.write(out);
	out.save(cout);					// 結果画像をセーブ
#if 1
      // CPUによって計算する．
	Profiler<>	profiler(1);
	Image<out_t>	outGold;
	for (u_int n = 0; n < 10; ++n)
	{
	    outGold = in;
	    profiler.start(0);
	  //op3x3(outGold.begin(), outGold.end(), OP<in_t, out_t>());
	    op3x3(outGold.begin(), outGold.end(), OP<in_t>());
	    profiler.stop().nextFrame();
	}
	profiler.print(cerr);
	outGold.save(cout);

      // 結果を比較する．
	const int	V = 160;
	for (u_int u = 0; u < out.width(); ++u)
	    cerr << ' ' << (out[V][u] - outGold[V][u]);
	cerr <<  endl;
#endif
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
