/*
 *  $Id: Cuda++.h,v 1.2 2009-04-21 23:30:35 ueshiba Exp $
 */
/*!
  \mainpage	libTUCuda++ - NVIDIA社のCUDAを利用するためのユティリティライブラリ
  \anchor	libTUCuda

  \section copyright 著作権
  平成14-21年（独）産業技術総合研究所 著作権所有

  創作者：植芝俊夫

  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
  等の行為を禁止します．
   
  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
  よび創作者は責任を負いません。

  Copyright 2002-2009.
  National Institute of Advanced Industrial Science and Technology (AIST)

  Creator: Toshio UESHIBA

  [AIST Confidential and all rights reserved.]
  This program is confidential. Any using, copying, changing or
  giving any information concerning with this program to others
  without permission by the copyright holder are strictly prohibited.

  [No Warranty.]
  The copyright holder or the creator are not responsible for any
  damages caused by using this program.

  \section abstract 概要
  libTUCuda++は，C++環境においてNVIDIA社のCUDAを利用するためのユティリティ
  ライブラリである．以下のようなクラスおよび関数が実装されている．

  <b>CUDAの初期化</b>
  - #TU::initializeCUDA(int, char*[])

  <b>デバイス側のグローバルメモリ領域にとられる1次元および2次元配列</b>
  - #TU::CudaDeviceMemory
  - #TU::CudaDeviceMemory2
  
  \file		Cuda++.h
  \brief	基本的なデータ型をグローバルな名前空間に追加
*/
#ifndef __TUCudaPP_h
#define __TUCudaPP_h

/*!
  \namespace	TU
  \brief	本ライブラリで定義されたクラスおよび関数を納める名前空間
*/
namespace TU
{
void	initializeCUDA(int argc, char* argv[]);
}

#endif	/* !__TUCudaPP_h */
