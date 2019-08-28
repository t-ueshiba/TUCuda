/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <cstdlib>
#include <limits>
#include "TU/io.h"
#include "TU/Rectify.h"
#include "TU/StereoUtility.h"

#define DEFAULT_PARAM_FILE	"stereo"
#define DEFAULT_CONFIG_DIRS	".:/usr/local/etc"
#define DEFAULT_SCALE		1.0

namespace TU
{
template <class T, class S> void
cudaJob(const Array2<T>& imageL, const Array2<T>& imageR, Array3<S>& costs,
	size_t winSize, size_t disparitySearchWidth, size_t intensityDiffMax);
template <class T, class S> void
cudaJob(const Array2<T>& imageL, const Array2<T>& imageR, Array2<S>& imageD,
	size_t winSize, size_t disparitySearchWidth, size_t disparityMax,
	size_t intensityDiffMax, size_t disparityInconsistency);

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
    
    Image<S>	imageD(rectify.width(0), rectify.height(0));
#if 1
  // コストを計算する．
    Array3<S>	costs;
    cudaJob(rectifiedImageL, rectifiedImageR, costs, params.windowSize,
	    params.disparitySearchWidth, params.intensityDiffMax);
    
  // ステレオマッチングを行う．
#  ifdef DISPARITY_MAJOR
    DisparitySelector<S>	selector(params.disparityMax,
					 params.disparityInconsistency);
    selector.initialize(costs.size<1>(), costs.size<2>());
    for (const auto& layer : costs)
	selector.accumulate(layer.cbegin(), layer.cend());
    selector.select(imageD.begin());
#  else
    std::copy(costs.cbegin(), costs.cend(),
	      make_range_iterator(make_matching_iterator<S>(
				      imageD.begin()->begin(),
				      params.disparitySearchWidth,
				      params.disparityMax,
				      params.disparityInconsistency,
				      params.doHorizontalBackMatch),
				  stride(imageD.begin()),
				  imageD.width()));
#  endif    
#else
  // コストを計算し，ステレオマッチングをを行う．
    cudaJob(rectifiedImageL, rectifiedImageR, imageD,
	    params.windowSize, params.disparitySearchWidth, params.disparityMax,
	    params.intensityDiffMax, params.disparityInconsistency);
#endif
    imageD.save(std::cout);
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
    size_t	intensityDiffMax	= 0;
    
  // コマンド行の解析．
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "p:c:w:d:m:i:")) != EOF; )
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
	  case 'i':
	    intensityDiffMax = atoi(optarg);
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
	    params.disparitySearchWidth	= disparitySearchWidth;
	if (intensityDiffMax != 0)
	    params.intensityDiffMax	= intensityDiffMax;

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
