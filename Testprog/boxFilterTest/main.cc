/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <stdexcept>
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/BoxFilter.h"

namespace TU
{
template <class S, class T> void
cudaJob(const Image<S>& in, Image<T>& out, size_t winSize)		;
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
	Image<out_t>	out;
	TU::cudaJob(in, out, winSize);
	out.save(cout);					// 結果画像をセーブ

      // CPUによって計算する．
	Profiler<>	profiler(1);
	Image<out_t>	outGold(in.width(), in.height());
	BoxFilter2	filter(winSize, winSize);
	for (u_int n = 0; n < 10; ++n)
	{
	    profiler.start(0);
	    filter.convolve(in.cbegin(), in.cend(), outGold.begin());
	    profiler.nextFrame();
	}
	profiler.print(cerr);
	outGold.save(cout);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
