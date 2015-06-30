/*
 *  $Id$
 */
#include <sstream>
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/CanvasPane.h"
#include "TU/v/CanvasPaneDC.h"
#include "TU/GuidedFilter.h"

namespace TU
{
typedef u_char	PixelType;

namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{c_WinSize, c_Regularization, c_Saturation, c_Cursor};

static int	range[][3] = {{1, 64, 1}, {0, 255, 1}, {1, 64, 4}};
static CmdDef	Cmds[] =
{
    {C_Slider, c_WinSize,	 11, "Window size:",	range[0], CA_None,
     0, 0, 1, 1, 0},
    {C_Slider, c_Regularization, 10, "Regularization:",	range[1], CA_None,
     1, 0, 1, 1, 0},
    {C_Slider, c_Saturation,     12, "Saturation:",     range[2], CA_None,
     0, 1, 1, 1, 0},
    {C_Label,  c_Cursor,	  0, "         ",	noProp,	  CA_None,
     1, 1, 1, 1, 0},
    EndOfCmds
};

/************************************************************************
*  class MyCanvasPane<T>						*
************************************************************************/
template <class T>
class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin, const Image<T>& image,
		 u_int width, u_int height)
	:CanvasPane(parentWin, width, height),
	 _dc(*this, width, height), _image(image)	{}

    virtual void	repaintUnderlay()		{ _dc << _image; }
    void		clear()				{ _dc.clear(); }
    void		drawPoint(int u, int v)
			{
			    _dc << foreground(BGR(255, 255, 0))
				<< Point2<int>(u, v);
			}
    void		setZoom(u_int mul, u_int div)
			{
			    _dc.setZoom(mul, div);
			}
    void		setSize(u_int width, u_int height)
			{
			    _dc.setSize(width, height, _dc.mul(), _dc.div());
			}
    virtual void	callback(CmdId id, CmdVal val)	;
    
  private:
    CanvasPaneDC	_dc;
    const Image<T>&	_image;
};

template <class T> void
MyCanvasPane<T>::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case Id_MouseMove:
      case Id_MouseButton1Press:
      case Id_MouseButton1Drag:
      case Id_MouseButton1Release:
      {
	CmdVal	logicalPosition(_dc.dev2logU(val.u), _dc.dev2logV(val.v));
	parent().callback(id, logicalPosition);
      }
        return;
    }

    parent().callback(id, val);
}
    
/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App& parentApp, const char* name,
		const Image<PixelType>& guide)				;

    void		showWeights(size_t u, size_t v)			;
    virtual void	callback(CmdId id, CmdVal val)			;

  private:
    const Image<PixelType>&	_guide;
    Image<float>		_weights;
    GuidedFilter2<float>	_gf2;
    CmdPane			_cmd;
    MyCanvasPane<PixelType>	_canvas;
    MyCanvasPane<float>		_weightsCanvas;
};

MyCmdWindow::MyCmdWindow(App& parentApp, const char* name,
			 const Image<PixelType>& guide)
    :CmdWindow(parentApp, name, 0, Colormap::RGBColor, 16, 0, 0),
     _guide(guide),
     _weights(),
     _gf2(5, 5, 1),
     _cmd(*this, Cmds),
     _canvas(*this, _guide, _guide.width(), _guide.height()),
     _weightsCanvas(*this, _weights, 256, 256)
{
    
    _cmd.place(0, 0, 2, 1);
    _canvas.place(0, 1, 1, 1);
    _weightsCanvas.place(1, 1, 1, 1);

    size_t	w = _cmd.getValue(c_WinSize);
    float	s = _cmd.getValue(c_Regularization).f();
    _gf2.setRowWinSize(w);
    _gf2.setColWinSize(w);
    _gf2.setEpsilon(s*s);
    colormap().setSaturationF(_cmd.getValue(c_Saturation).f());
    _weightsCanvas.setZoom(4, 1);
    _weightsCanvas.setSize(2*w - 1, 2*w - 1);

    show();
}

void
MyCmdWindow::showWeights(size_t u, size_t v)
{
    Image<float>	in(_guide.width(), _guide.height()),
			out(_guide.width(), _guide.height());
    size_t		w = _cmd.getValue(c_WinSize);

    in[v][u] = 255;
    _gf2.convolve(in.begin(), in.end(),
		  _guide.begin(), _guide.end(), out.begin());

    size_t	uc = (u < w -1 ? w - 1 :
		      u > out.width() - w + 1 ? out.width() - w + 1 : u);
    size_t	vc = (v < w -1 ? w - 1 :
		      v > out.height() - w + 1 ? out.height() - w + 1 : v);
    _weights = out(uc - w + 1, vc - w + 1, 2*w - 1, 2*w - 1);
    _weightsCanvas.setSize(2*w - 1, 2*w - 1);
    _weightsCanvas.repaintUnderlay();
}
    
void
MyCmdWindow::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case M_Exit:
	app().exit();
	break;

      case c_WinSize:
	_gf2.setRowWinSize(val);
	_gf2.setColWinSize(val);
	break;

      case c_Regularization:
	_gf2.setEpsilon(val.f()*val.f());
	break;
	
      case c_Saturation:
	colormap().setSaturationF(val.f());
	_weightsCanvas.repaintUnderlay();
	break;
	
      case Id_MouseButton1Press:
      case Id_MouseButton1Drag:
	showWeights(val.u, val.v);
      case Id_MouseMove:
      {
	int	w = _cmd.getValue(c_WinSize);
	_weightsCanvas.drawPoint(w - 1, w - 1);
	  
	std::ostringstream	s;
	s << '(' << val.u << ',' << val.v << ')';
	_cmd.setString(c_Cursor, s.str().c_str());
      }
        break;

      case Id_MouseButton1Release:
	_weightsCanvas.clear();
	break;
    }
}

}
}

int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    try
    {
	v::App		vapp(argc, argv);
	Image<u_char>	guide;
	guide.restore(cin);

      // GUIオブジェクトを作り，イベントループを起動．
	v::MyCmdWindow	myWin(vapp, "Weights of guided filter", guide);
	vapp.run();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}