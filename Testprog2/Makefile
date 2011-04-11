#
#  $Id: Makefile,v 1.6 2011-04-11 08:06:06 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I.. -I$(PREFIX)/include -I$(CUDAHOME)/include -I$(CUDASDK)/common/inc

PROGRAM		= test1394cuda
LIBS		= -lTUIeee1394++ -lTUTools++ -lraw1394
ARCH		= $(shell uname -m)

CCC		= g++
CPPFLAGS	= -DHAVE_LIBTUTOOLS__
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O # -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS     += -axP
  else
    CPPFLAGS   += -DSSSE3
#    CFLAGS     += -xN
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(NVCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC
EXTHDRS		= ../TU/CudaArray++.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Ieee1394++.h \
		/usr/local/include/TU/Ieee1394CameraArray.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Normalize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/io.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/TU/utility.h
HDRS		= cuda_image_interpolate_kernel.h
SRCS		= cuda_image_interpolate.cu \
		main.cc
OBJS		= cuda_image_interpolate.o \
		main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.6 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
cuda_image_interpolate.o: ../TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/utility.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Normalize.h /usr/local/include/TU/Minimize.h \
	cuda_image_interpolate_kernel.h
main.o: /usr/local/include/TU/Ieee1394CameraArray.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Ieee1394++.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/utility.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/io.h
