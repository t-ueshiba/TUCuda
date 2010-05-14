#
#  $Id: Makefile,v 1.4 2010-05-14 00:38:24 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include -I$(CUDASDK)/common/inc -I$(CUDAHOME)/include

PROGRAM		= test1394cuda
LIBS		= -lTUIeee1394++ -lTUTools++ -lTUCuda++ -lraw1394
ARCH		= $(shell uname -m)
ifeq ($(ARCH), x86_64)
  LIBS	       += -lcutil_x86_64
else
  LIBS	       += -lcutil_x86_i386
endif

CPPFLAGS	= -DHAVE_LIBTUTOOLS__
CFLAGS		= -g -O
NVCCFLAGS	= -g -O
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
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Cuda++.h \
		/usr/local/include/TU/CudaDeviceMemory.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Ieee1394++.h \
		/usr/local/include/TU/Ieee1394CameraArray.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Normalize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/types.h
HDRS		= cuda_image_interpolate_kernel.h
SRCS		= cuda_image_interpolate.cu \
		main.cc
OBJS		= cuda_image_interpolate.o \
		main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.4 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
cuda_image_interpolate.o: /usr/local/include/TU/CudaDeviceMemory.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Normalize.h \
	cuda_image_interpolate_kernel.h
main.o: /usr/local/include/TU/Cuda++.h \
	/usr/local/include/TU/Ieee1394CameraArray.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Ieee1394++.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Normalize.h
