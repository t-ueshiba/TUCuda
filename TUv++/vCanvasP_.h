/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id$  
 */
#ifndef __VCANVASP_H
#define __VCANVASP_H

#include "vCanvas_.h"

/************************************************************************
*  Private definitions for vCanvas widget				*
************************************************************************/
/* New fields for the vCanvas widget class record */
typedef struct _VCanvasClass
{
    int			empty;		/* need something */
} VCanvasClassPart;

/* Full class record declaration */
typedef struct _VCanvasClassRec
{
    CoreClassPart	core_class;
    VCanvasClassPart	vcanvas_class;
} VCanvasClassRec;

extern VCanvasClassRec vCanvasClassRec;

/* New fields for the vCanvas widget record */
typedef struct
{
    int			backing_store;	 /* Whether we allow backing store */
    XtCallbackList	ginit_callbacks; /* called once when realized	   */
} VCanvasPart;

/************************************************************************
*  Full instance record declaration					*
************************************************************************/
typedef struct _VCanvasRec
{
    CorePart		core;
    VCanvasPart		vcanvas;
} VCanvasRec;

#endif	/* !__VCANVASP_H	*/
