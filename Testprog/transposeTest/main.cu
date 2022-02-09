/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cu/Array++.h"
#include "TU/cu/algorithm.h"
#include "TU/cu/chrono.h"

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;

  //using	in_t  = u_char;
    using	in_t  = float;
  //using	out_t = u_char;
    using	out_t = float;
    
    try
    {
	Image<in_t>	in;
	in.restore(cin);				// 原画像を読み込む
	in.save(cout);					// 原画像をセーブ

      // GPUによって計算する．
	cu::Array2<in_t>	in_d(in);
	cu::Array2<out_t>	out_d(in_d.ncol(), in_d.nrow());

	cu::transpose(in_d.cbegin(), in_d.cend(), out_d.begin());
	cudaDeviceSynchronize();

	Profiler<cu::clock>	cuProfiler(1);
	constexpr size_t	NITER = 1000;
	for (size_t n = 0; n < NITER; ++n)
	{
	    cuProfiler.start(0);
	    cu::transpose(in_d.cbegin(), in_d.cend(), out_d.begin());
	    cuProfiler.nextFrame();
	}
	cuProfiler.print(std::cerr);
	
	Image<out_t>	out(out_d);
	out.save(cout);					// 結果画像をセーブ
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
