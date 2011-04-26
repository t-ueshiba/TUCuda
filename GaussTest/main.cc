/*
 *  $Id: main.cc,v 1.2 2011-04-26 04:53:44 ueshiba Exp $
 */
#include <stdexcept>
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/IIRFilterMT.h"
#include "TU/GaussianConvolver.h"
#include "filterImageGold.h"
#include "TU/CudaGaussianConvolver.h"
#include <cuda_runtime.h>
#include <cutil.h>

namespace TU
{
/************************************************************************
*  static fucntions							*
************************************************************************/
Array<float>
computeGaussianCoefficients(float sigma, u_int lobeSize)
{
    using namespace	std;

    Array<float>	coeff(lobeSize + 1);
    for (u_int i = 0; i <= lobeSize; ++i)
    {
	float	dx = float(lobeSize - i) / sigma;
	coeff[i] = exp(-0.5 * dx * dx);
    }

    float	sum = coeff[lobeSize];
    for (u_int i = 0; i < lobeSize; ++i)
	sum += (2.0f * coeff[i]);
    
    for (u_int i = 0; i <= lobeSize; ++i)
	coeff[i] /= sum;

  //#ifdef _DEBUG
    sum = coeff[lobeSize];
    for (u_int i = 0; i < lobeSize; ++i)
    {
	sum += 2*coeff[i];
	cerr << coeff[i] << endl;
    }
    cerr << coeff[lobeSize] << endl;
    cerr << "Sum of coeeficients = " << sum << endl;
  //#endif
    return coeff;
}
    
}

/************************************************************************
*  Global fucntions							*
************************************************************************/
#define CONVOLVE	smooth

int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;

  //typedef u_char	in_t;
    typedef float	in_t;
  //typedef u_char	out_t;
    typedef float	out_t;
    
    float		sigma	 = 1.0;
    u_int		lobeSize = 16;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "s:l:")) != -1; )
	switch (c)
	{
	  case 's':
	    sigma = atof(optarg);
	    break;
	  case 'l':
	    lobeSize = atoi(optarg);
	    break;
	}
    
    try
    {
	Array<float>	coeff = computeGaussianCoefficients(sigma, lobeSize);
	Image<in_t>	in;
	in.restore(cin);				// 原画像を読み込む
	in.save(cout);				// 原画像をセーブ

      // GPUによって計算する．
	CudaGaussianConvolver2	cudaFilter(sigma);

	CudaArray2<in_t>	in_d(in);
	CudaArray2<out_t>	out_d;

	u_int		timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timer));		// タイマーを作成
	cudaFilter.CONVOLVE(in_d, out_d);		// warm-up
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#if 0
	CUT_SAFE_CALL(cutStartTimer(timer));
	u_int	NITER = 1000;
	for (u_int n = 0; n < NITER; ++n)
	    cudaFilter.CONVOLVE(in_d, out_d);		// フィルタをかける
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(timer));

	cerr << float(NITER * 1000) / cutGetTimerValue(timer) << "fps" << endl;
	CUT_SAFE_CALL(cutDeleteTimer(timer));		// タイマーを消去
#endif
	Image<out_t>	out;
	out_d.write(out);
	out.save(cout);					// 結果画像をセーブ

      // CPUによって計算する．
	Profiler	profiler(1);
	Image<out_t>	outGold;
#if 0
	for (u_int n = 0; n < 10; ++n)
	{
	    profiler.start(0);
	    filterImageGold(in, outGold, coeff);
	    profiler.stop().nextFrame();
	}
#else
	GaussianConvolver2<>	convolver(sigma);
      //GaussianConvolver2<BilateralIIRFilterThreadArray<4u, Image<float>, Array2<Array<float> > >, BilateralIIRFilterThreadArray<4u, Array2<Array<float> >, Image<float> > >	convolver(sigma, 8);
	for (u_int n = 0; n < 10; ++n)
	{
	    profiler.start(0);
	    convolver.CONVOLVE(in, outGold);
	    profiler.stop().nextFrame();
	}
#endif
	profiler.print(cerr);
	outGold.save(cout);

      // 結果を比較する．
	for (u_int u = lobeSize; u < out.width() - lobeSize; ++u)
	    cerr << ' ' << 100*(out[240][u] - outGold[240][u])
			      / abs(outGold[240][u]);
	cerr <<  endl;

    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
