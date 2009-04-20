/*
 *  $Id: main.cc,v 1.1 2009-04-20 01:17:39 ueshiba Exp $
 */
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <stdexcept>
#include "TU/Cuda++.h"
#include "TU/Ieee1394++.h"

#define DEFAULT_CONFIG_DIRS	".:/usr/local/etc/cameras"
#define DEFAULT_CAMERA_NAME	"IEEE1394Camera"

#define PIXEL_TYPE		ImageBase::U_CHAR

namespace TU
{
void	interpolate(const Array2<ImageLine<RGBA> >& image0,
		    const Array2<ImageLine<RGBA> >& image1,
			  Array2<ImageLine<RGBA> >& image2);

/************************************************************************
*  class CameraArray							*
************************************************************************/
class CameraArray : public Array<Ieee1394Camera*>
{
  public:
    CameraArray(std::istream& in, bool i1394b, u_int ncammax)		;
    ~CameraArray()							;
};

CameraArray::CameraArray(std::istream& in, bool i1394b, u_int ncammax)
    :Array<Ieee1394Camera*>()
{
    using namespace	std;

    u_int	delay, ncameras;
    in >> delay >> ncameras;		// Read delay value and #cameras.
    if (ncammax != 0 && ncammax < ncameras)
	ncameras = ncammax;
    resize(ncameras);
    
    for (int i = 0; i < dim(); ++i)
    {
	string	s;
	in >> s;			// Read global unique ID.
	u_int64	uniqId = strtoull(s.c_str(), 0, 0);
	(*this)[i] = new Ieee1394Camera(Ieee1394Camera::Monocular,
					i1394b, uniqId, delay);

	in >> *(*this)[i];		// Read camera parameters.
    }
}

CameraArray::~CameraArray()
{
    for (int i = 0; i < dim(); ++i)
	delete (*this)[i];
}

/************************************************************************
*  static functions							*
************************************************************************/
static std::string
openFile(std::ifstream& in, const std::string& dirs, const std::string& name)
{
    using namespace		std;

    string::const_iterator	p = dirs.begin();
    do
    {
	string::const_iterator	q = find(p, dirs.end(), ':');
	string			fullName = string(p, q) + '/' + name;
	in.open(fullName.c_str());
	if (in)
	    return fullName;
	p = q;
    } while (p++ != dirs.end());

    throw runtime_error("Cannot open input file \"" + name +
			"\" in \"" + dirs + "\"!!");
    return string();
}

static bool	active = true;

static void
handler(int sig)
{
    using namespace	std;
    
    cerr << "Signal [" << sig << "] caught!" << endl;
    active = false;
}

template <class T> static void
doJob(const CameraArray& cameras)
{
    using namespace	std;
    
  // Set signal handler.
    signal(SIGINT,  handler);
    signal(SIGPIPE, handler);

  // 1�ե졼�ढ����β������Ȥ��Υե����ޥåȤ���ϡ�
    Array<Image<T> >	images(cameras.dim() + 1);
    cout << images.dim() << endl;
    for (int i = 0; i < images.dim(); ++i)
    {
	images[i].resize(cameras[0]->height(), cameras[0]->width());
	images[i].saveHeader(cout, PIXEL_TYPE);
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
	    *cameras[i] >> images[i];			// �����ΰ�ؤ�ž��

	interpolate(images[0], images[1], images[2]);

	for (int i = 0; i < images.dim(); ++i)
	    if (!images[i].saveData(cout, PIXEL_TYPE))	// stdout�ؤν���
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
    
    initializeCUDA(argc, argv);

    using namespace	std;
    using namespace	TU;
    
    string		configDirs = DEFAULT_CONFIG_DIRS;
    string		cameraName = DEFAULT_CAMERA_NAME;
    bool		i1394b	   = false;
    u_int		ncammax	   = 2;
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
	    ncammax = atoi(optarg);
	    break;
	}
    
    try
    {
      // IEEE1394�����Υ����ץ�
	ifstream	in;
	openFile(in, configDirs, cameraName + ".conf");
	CameraArray	cameras(in, i1394b, ncammax);

	for (int i = 0; i < cameras.dim(); ++i)
	    cerr << "camera " << i << ": uniqId = "
		 << hex << setw(16) << setfill('0')
		 << cameras[i]->globalUniqueId() << dec << endl;

      // �����Υ���ץ���Ƚ��ϡ�
	doJob<RGBA>(cameras);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
    }

    return 0;
}
