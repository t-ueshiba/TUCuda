#
#  $Id: Makefile,v 1.8 2012-01-23 00:22:45 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I.. -I$(PREFIX)/include -I$(CUDAHOME)/include -I$(CUDASDK)/common/inc

PROGRAM		= cuda_image_interpolate
LIBS		= -lTUTools++ -lcutil_$(shell uname -m)

CPPFLAGS	=
CFLAGS		= -g -O
NVCCFLAGS	= -g -O
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O # -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
  else
    CPPFLAGS   += -DSSSE3
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
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/TU/utility.h
HDRS		=
SRCS		= cuda_image_interpolate.cu \
		main.cc
OBJS		= cuda_image_interpolate.o \
		main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.8 $	|		\
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
	/usr/local/include/TU/Minimize.h
main.o: /usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/utility.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Minimize.h
