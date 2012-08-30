/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <stdexcept>
#include "TU/Ieee1394CameraArray.h"

#define DEFAULT_CONFIG_DIRS	".:/usr/local/etc/cameras"
#define DEFAULT_CAMERA_NAME	"IEEE1394Camera"

namespace TU
{
template <class T> void
interpolate(const Image<T>& image0, const Image<T>& image1, Image<T>& image2);

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
doJob(const Ieee1394CameraArray& cameras)
{
    using namespace	std;
    
  // Set signal handler.
    signal(SIGINT,  handler);
    signal(SIGPIPE, handler);

  // 1�ե졼�ढ����β������Ȥ��Υե����ޥåȤ���ϡ�
    Array<Image<T> >	images(cameras.dim() + 1);
    cout << 'M' << images.dim() << endl;
    for (int i = 0; i < images.dim(); ++i)
    {
	images[i].resize(cameras[0]->height(), cameras[0]->width());
	images[i].saveHeader(cout);
    }
    
  // �������Ϥγ��ϡ�
    for (int i = 0; i < cameras.dim(); ++i)
	cameras[i]->continuousShot();

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

	for (int i = 0; i < cameras.dim(); ++i)
	    cameras[i]->snap();				// ����
	for (int i = 0; i < cameras.dim(); ++i)
	    *cameras[i] >> images[i];			// �絭���ؤ�ž��

	interpolate(images[0], images[1], images[2]);	// CUDA�ˤ�����

	for (int i = 0; i < images.dim(); ++i)
	    if (!images[i].saveData(cout))		// stdout�ؤν���
		active = false;
    }

  // �������Ϥ���ߡ�
    for (int i = 0; i < cameras.dim(); ++i)
	cameras[i]->stopContinuousShot();
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
    
    const char*		configDirs = DEFAULT_CONFIG_DIRS;
    const char*		cameraName = DEFAULT_CAMERA_NAME;
    bool		i1394b	   = false;
    int			ncameras   = 2;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "d:c:Bn:")) != EOF; )
	switch (c)
	{
	  case 'd':
	    configDirs = optarg;
	    break;
	  case 'c':
	    cameraName = optarg;
	    break;
	  case 'B':
	    i1394b = true;
	    break;
	  case 'n':
	    ncameras = atoi(optarg);
	    break;
	}
    
    try
    {
      // IEEE1394�����Υ����ץ�
	Ieee1394CameraArray	cameras(cameraName, configDirs,
					i1394b, ncameras);
	if (cameras.dim() == 0)
	    return 0;

	for (int i = 0; i < cameras.dim(); ++i)
	    cerr << "camera " << i << ": uniqId = "
		 << hex << setw(16) << setfill('0')
		 << cameras[i]->globalUniqueId() << dec << endl;

      // �����Υ���ץ���Ƚ��ϡ�
	doJob<u_char>(cameras);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
    }

    return 0;
}
