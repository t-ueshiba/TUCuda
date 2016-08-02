/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <stdexcept>
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/algorithm.h"
#include "TU/cuda/functional.h"

//#define OPERATOR	cuda::det3x3
//#define OPERATOR	cuda::laplacian3x3
//#define OPERATOR	cuda::sobelAbs3x3
#define OPERATOR	cuda::maximal3x3
//#define OPERATOR	cuda::minimal3x3

namespace TU
{
template <template <class> class OP, class S, class T> void
cudaJob(const Image<S>& in, Image<T>& out)				;
}

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

	Image<out_t>	out;
	TU::cudaJob<OPERATOR>(in, out);
	out.save(cout);					// 結果画像をセーブ
#if 1
      // CPUによって計算する．
	Profiler<>	profiler(1);
	Image<out_t>	outGold;
	for (u_int n = 0; n < 10; ++n)
	{
	    outGold = in;
	    profiler.start(0);
	    op3x3(outGold.begin(), outGold.end(), OPERATOR<in_t>());
	    profiler.nextFrame();
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
