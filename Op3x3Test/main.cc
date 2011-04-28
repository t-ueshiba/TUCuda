/*
 *  $Id: main.cc,v 1.2 2011-04-28 07:59:22 ueshiba Exp $
 */
#include <stdexcept>
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/utility.h"
#include "TU/CudaUtility.h"
#include "TU/GaussianConvolver.h"
#include <cuda_runtime.h>
#include <cutil.h>

//#define OP	det3x3
//#define OP	laplacian3x3
//#define OP	sobelAbs3x3
#define OP	maximal3x3
//#define OP	minimal3x3

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
    
    float		sigma = 1.0;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "s:")) != -1; )
	switch (c)
	{
	  case 's':
	    sigma = atof(optarg);
	    break;
	}
    
    try
    {
	Image<in_t>	in;
	in.restore(cin);				// 原画像を読み込む
	in.save(cout);					// 原画像をセーブ

      // GPUによって計算する．
	CudaArray2<in_t>	in_d(in);
	CudaArray2<out_t>	out_d;

	u_int		timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timer));		// タイマーを作成
      //cudaOp3x3(in_d, out_d, OP<in_t, out_t>());	// warm-up
	cudaOp3x3(in_d, out_d, OP<in_t>());		// warm-up
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUT_SAFE_CALL(cutStartTimer(timer));
	u_int	NITER = 1000;
	for (u_int n = 0; n < NITER; ++n)
	  //cudaOp3x3(in_d, out_d, OP<in_t, out_t>());	// フィルタリング
	    cudaOp3x3(in_d, out_d, OP<in_t>());		// フィルタリング
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(timer));

	cerr << float(NITER * 1000) / cutGetTimerValue(timer) << "fps" << endl;
	CUT_SAFE_CALL(cutDeleteTimer(timer));		// タイマーを消去

	Image<out_t>	out;
	out_d.write(out);
	out.save(cout);					// 結果画像をセーブ
#if 1
      // CPUによって計算する．
	Profiler	profiler(1);
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
