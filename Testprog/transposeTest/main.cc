/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <stdexcept>
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/algorithm.h"

namespace TU
{
template <class S, class T> void
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
	TU::cudaJob(in, out);
	out.save(cout);					// 結果画像をセーブ
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
