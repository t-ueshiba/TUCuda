/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <cstdlib>
#include <signal.h>
#include <sys/time.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <stdexcept>
#include "TU/IIDCCameraArray.h"
#include "TU/cu/Array++.h"

namespace TU
{
namespace cu
{
template <class T> void
interpolate(const Array2<T>& d_image0,
	    const Array2<T>& d_image1, Array2<T>& d_image2);
}

/************************************************************************
*  static functions							*
************************************************************************/
static bool	active = true;

static void
handler(int sig)
{
    using namespace	std;
    
    cerr << "Signal [" << sig << "] caught!" << endl;
    active = false;
}

template <class T> static void
doJob(IIDCCameraArray& cameras)
{
    using namespace	std;
    
  // Set signal handler.
    signal(SIGINT,  handler);
    signal(SIGPIPE, handler);

  // 1フレームあたりの画像数とそのフォーマットを出力．
    Array<Image<T> >	images(cameras.size() + 1);
    cout << 'M' << images.size() << endl;
    for (size_t i = 0; i < images.size(); ++i)
    {
	images[i].resize(cameras[0].height(), cameras[0].width());
	images[i].saveHeader(cout);
    }

  // デバイス画像の確保
    cu::Array2<T>	d_images[3];
    
  // カメラ出力の開始．
    for (size_t i = 0; i < cameras.size(); ++i)
	cameras[i].continuousShot(true);

    int		nframes = 0;
    timeval	start;
    while (active)
    {
	if (nframes == 10)
	{
	    timeval      end;
	    gettimeofday(&end, NULL);
	    double	interval = (end.tv_sec  - start.tv_sec) +
				   (end.tv_usec - start.tv_usec) / 1.0e6;
	    cerr << nframes / interval << " frames/sec" << endl;
	    nframes = 0;
	}
	if (nframes++ == 0)
	    gettimeofday(&start, NULL);

	for (size_t i = 0; i < cameras.size(); ++i)
	    cameras[i].snap();				// 撮影
	for (size_t i = 0; i < cameras.size(); ++i)
	    cameras[i] >> images[i];			// 主記憶への転送
	for (size_t i = 0; i < cameras.size(); ++i)
	    d_images[i] = images[i];			// デバイスへの転送

	cu::interpolate(d_images[0],
			d_images[1], d_images[2]);	// CUDAによる補間
	images[2] = d_images[2];			// ホストへの転送

	for (size_t i = 0; i < images.size(); ++i)
	    if (!images[i].saveData(cout))		// stdoutへの出力
		active = false;
    }

  // カメラ出力の停止．
    for (size_t i = 0; i < cameras.size(); ++i)
	cameras[i].continuousShot(false);
}

}
/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;
    
    const char*		cameraName = IIDCCameraArray::DEFAULT_CAMERA_NAME;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "c:")) != EOF; )
	switch (c)
	{
	  case 'c':
	    cameraName = optarg;
	    break;
	}
    
    try
    {
      // IIDCカメラのオープン．
	IIDCCameraArray	cameras(cameraName);
	cameras.restore();
	
	if (cameras.size() == 0)
	    return 0;

	for (size_t i = 0; i < cameras.size(); ++i)
	    cerr << "camera " << i << ": uniqId = "
		 << hex << setw(16) << setfill('0')
		 << cameras[i].globalUniqueId() << dec << endl;

      // 画像のキャプチャと出力．
	doJob<u_char>(cameras);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
    }

    return 0;
}
