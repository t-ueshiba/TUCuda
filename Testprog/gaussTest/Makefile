#
#  $Id: Makefile,v 1.3 2012-08-30 12:25:01 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I../filterTest -I$(PREFIX)/include -I$(CUDAHOME)/include -I$(CUDASDK)/common/inc

PROGRAM		= $(shell basename $(PWD))
LIBS		= -lTUTools++ -lTUCuda++ -ltbb
ifeq ($(OSTYPE), darwin)
  LIBS	       += -lcutil_x86_64 -lsvml
else
  LIBS	       += -lcutil_$(shell uname -m)
endif

CPPFLAGS	= -DUSE_TBB
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LDFLAGS		= -L$(CUDASDK)/lib
LINKER		= $(NVCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/CudaArray++.h \
		/usr/local/include/TU/CudaFilter.h \
		/usr/local/include/TU/CudaGaussianConvolver.h \
		/usr/local/include/TU/GaussianConvolver.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/IIRFilter.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Profiler.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

include $(PROJECT)/lib/p.mk

###
main.o: /usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/Profiler.h \
	/usr/local/include/TU/GaussianConvolver.h \
	/usr/local/include/TU/IIRFilter.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/CudaGaussianConvolver.h \
	/usr/local/include/TU/CudaFilter.h \
	/usr/local/include/TU/CudaArray++.h
