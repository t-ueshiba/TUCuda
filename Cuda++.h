/*
 *  $Id: Cuda++.h,v 1.2 2009-04-21 23:30:35 ueshiba Exp $
 */
/*!
  \mainpage	libTUCuda++ - NVIDIA�Ђ�CUDA�𗘗p���邽�߂̃��e�B���e�B���C�u����
  \anchor	libTUCuda

  \section copyright ���쌠
  ����14-21�N�i�Ɓj�Y�ƋZ�p���������� ���쌠���L

  �n��ҁF�A�ŏr�v

  �{�v���O�����́i�Ɓj�Y�ƋZ�p�����������̐E���ł���A�ŏr�v���n�삵�C
  �i�Ɓj�Y�ƋZ�p���������������쌠�����L����閧���ł��D���쌠���L
  �҂ɂ�鋖�Ȃ��ɖ{�v���O�������g�p�C�����C���ρC��O�҂֊J������
  ���̍s�ׂ��֎~���܂��D
   
  ���̃v���O�����ɂ���Đ����邢���Ȃ鑹�Q�ɑ΂��Ă��C���쌠���L�҂�
  ��ёn��҂͐ӔC�𕉂��܂���B

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

  \section abstract �T�v
  libTUCuda++�́CC++���ɂ�����NVIDIA�Ђ�CUDA�𗘗p���邽�߂̃��e�B���e�B
  ���C�u�����ł���D�ȉ��̂悤�ȃN���X����ъ֐�����������Ă���D

  <b>CUDA�̏�����</b>
  - #TU::initializeCUDA(int, char*[])

  <b>�f�o�C�X���̃O���[�o���������̈�ɂƂ���1���������2�����z��</b>
  - #TU::CudaDeviceMemory
  - #TU::CudaDeviceMemory2
  
  \file		Cuda++.h
  \brief	��{�I�ȃf�[�^�^���O���[�o���Ȗ��O��Ԃɒǉ�
*/
#ifndef __TUCudaPP_h
#define __TUCudaPP_h

/*!
  \namespace	TU
  \brief	�{���C�u�����Œ�`���ꂽ�N���X����ъ֐���[�߂閼�O���
*/
namespace TU
{
void	initializeCUDA(int argc, char* argv[]);
}

#endif	/* !__TUCudaPP_h */
