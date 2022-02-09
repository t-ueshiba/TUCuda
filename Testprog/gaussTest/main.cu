/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/GaussianConvolver.h"
#include "TU/Profiler.h"
#include "TU/cu/FIRGaussianConvolver.h"
#include "TU/cu/chrono.h"
#include "TU/cu/vec.h"
#include "filterImageGold.h"

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

    using in_t	= u_char;
    using mid_t	= float;
    using out_t	= u_char;
    // using in_t	= RGB;
    // using mid_t	= float3;
    // using out_t	= RGB;
    
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
	Image<in_t>	in;
	in.restore(cin);				// 原画像を読み込む
	in.save(cout);					// 原画像をセーブ

      // GPUによって計算する．
	cu::FIRGaussianConvolver2<mid_t>	cudaFilter(sigma);
	cu::Array2<in_t>			in_d(in);
	cu::Array2<out_t>			out_d(in_d.nrow(),
						      in_d.ncol());
	cudaFilter.CONVOLVE(in_d.cbegin(), in_d.cend(), out_d.begin(), true);
	cudaDeviceSynchronize();

	Profiler<cu::clock>	cuProfiler(1);
	constexpr size_t	NITER = 1000;

	for (size_t n = 0; n < NITER; ++n)
	{
	    cuProfiler.start(0);
	    cudaFilter.CONVOLVE(in_d.cbegin(), in_d.cend(),
				out_d.begin(), true);
	    cuProfiler.nextFrame();
	}
	cuProfiler.print(cerr);

	Image<out_t>	out(out_d);
	out.save(cout);					// 結果画像をセーブ

      // CPUによって計算する．
	Profiler<>	profiler(1);
	Image<out_t>	outGold(in.width(), in.height());
#if 0
	Array<float>	coeff = computeGaussianCoefficients(sigma, lobeSize);
	for (u_int n = 0; n < 10; ++n)
	{
	    profiler.start(0);
	    filterImageGold(in, outGold, coeff);
	    profiler.nextFrame();
	}
#else
	GaussianConvolver2<float>	convolver(sigma);
	for (u_int n = 0; n < 10; ++n)
	{
	    profiler.start(0);
	    convolver.smooth(in.begin(), in.end(), outGold.begin());
	    profiler.nextFrame();
	}
#endif
	profiler.print(cerr);
	outGold.save(cout);

      // 結果を比較する．
	// for (u_int u = lobeSize; u < out.width() - lobeSize; ++u)
	//     cerr << ' ' << 100*(out[240][u] - outGold[240][u])
	// 		      / abs(outGold[240][u]);
	// cerr <<  endl;

    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
