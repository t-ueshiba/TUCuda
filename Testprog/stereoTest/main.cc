/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <cstdlib>
#include "TU/algorithm.h"
#include "TU/io.h"
#include "TU/Rectify.h"
#include "TU/StereoUtility.h"
#include "TU/Profiler.h"
#if 1
#  include "TU/cuda/BoxFilter.h"
#else
#  include "TU/cuda/NewBoxFilter.h"
#endif
#include "TU/cuda/functional.h"
#include "TU/cuda/chrono.h"

#define DEFAULT_PARAM_FILE	"stereo"
#define DEFAULT_CONFIG_DIRS	".:/usr/local/etc/cameras"
#define DEFAULT_SCALE		1.0

namespace TU
{
template <class T, class S> void
cudaJob(const Array2<T>& imageL, const Array2<T>& imageR,
	Array3<S>& scores, size_t winSize, size_t disparitySearchWidth)	;

/************************************************************************
*  static functions							*
************************************************************************/
template <class S, class T> static void
doJob(const Image<T>& imageL,
      const Image<T>& imageR, const StereoParameters& params)
{
  // 画像を平行化する．
    Rectify	rectify;
    Image<T>	rectifiedImageL, rectifiedImageR;
    rectify.initialize(imageL, imageR, 1.0,
		       params.disparitySearchWidth, params.disparityMax);
    rectify(imageL, imageR, rectifiedImageL, rectifiedImageR);

    std::cerr << "Left image:  "
	      << rectifiedImageL.height() << "(H)x"
	      << rectifiedImageL.width()  << "(W)"
	      << std::endl;
    std::cerr << "Right image: "
	      << rectifiedImageR.height() << "(H)x"
	      << rectifiedImageR.width()  << "(W)"
	      << std::endl;
    
  // スコアを計算する．
    Array3<S>	scores;
    cudaJob(rectifiedImageL, rectifiedImageR, scores,
	    params.windowSize, params.disparitySearchWidth);
    
  // ステレオマッチングを行う．
    Image<S>	disparityMap(rectify.width(0), rectify.height(0));
    const auto	rowD = make_range_iterator(
			   make_matching_iterator<S>(
			       disparityMap.begin()->begin(),
			       params.disparitySearchWidth,
			       params.disparityMax,
			       params.disparityInconsistency,
			       params.doHorizontalBackMatch),
			   stride(disparityMap.begin()),
			   disparityMap.width());
    std::copy(scores.cbegin(), scores.cend(), rowD);

  //rectifiedImagesL.save(std::cout);
  //rectifiedImagesR.save(std::cout);
    disparityMap.save(std::cout);
}

}
/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	TU;

    using pixel_type	= u_char;
  //using score_type	= short;
    using score_type	= float;
    
    std::string	paramFile		= DEFAULT_PARAM_FILE;
    std::string	configDirs		= DEFAULT_CONFIG_DIRS;
    size_t	windowSize		= 0;
    size_t	disparitySearchWidth	= 0;
    size_t	disparityMax		= 0;
    
  // コマンド行の解析．
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "p:c:w:d:m:")) != EOF; )
	switch (c)
	{
	  case 'p':
	    paramFile = optarg;
	    break;
	  case 'c':
	    configDirs = optarg;
	    break;
	  case 'w':
	    windowSize = atoi(optarg);
	    break;
	  case 'd':
	    disparitySearchWidth = atoi(optarg);
	    break;
	  case 'm':
	    disparityMax = atoi(optarg);
	    break;
	}
    
  // 本当のお仕事．
    try
    {
	std::ifstream		in;
	openFile(in, paramFile, configDirs, ".params");
	StereoParameters	params;
	params.get(in);
	    
	if (windowSize != 0)
	    params.windowSize		= windowSize;
	if (disparityMax != 0)
	    params.disparityMax		= disparityMax;
	if (disparitySearchWidth != 0)
	    params.disparitySearchWidth = disparitySearchWidth;

      // ステレオマッチングパラメータを表示．
	std::cerr << "--- Stereo matching parameters ---\n";
	params.put(std::cerr);

      // 画像を読み込む．
	Image<pixel_type>	imageL, imageR;
	imageL.restore(std::cin);
	imageR.restore(std::cin);

	doJob<score_type>(imageL, imageR, params);
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
