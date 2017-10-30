#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
#PROGRAM		= $(shell basename $(PWD))
LIBRARY		= lib$(shell basename $(PWD))

IDLDIR		= .
IDLS		=

INCDIRS		= -I. -I$(PREFIX)/include -I$(CUDAHOME)/include
CPPFLAGS	= -DNDEBUG #-DSSE4
CFLAGS		= -O3
NVCCFLAGS	= -O3
CCFLAGS		= $(CFLAGS)

LIBS		=

LINKER		= $(NVCC)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/algorithm.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/range.h \
		/usr/local/include/TU/tuple.h
HDRS		= TU/cuda/Array++.h \
		TU/cuda/BoxFilter.h \
		TU/cuda/FIRFilter.h \
		TU/cuda/FIRGaussianConvolver.h \
		TU/cuda/Texture.h \
		TU/cuda/algorithm.h \
		TU/cuda/allocator.h \
		TU/cuda/chrono.h \
		TU/cuda/functional.h
SRCS		= FIRFilter.cu \
		FIRGaussianConvolver.cc \
		chrono.cc
OBJS		= FIRFilter.o \
		FIRGaussianConvolver.o \
		chrono.o

#include $(PROJECT)/lib/rtc.mk		# modified: CPPFLAGS, LIBS
#include $(PROJECT)/lib/cnoid.mk	# modified: CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# added:    PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
FIRFilter.o: TU/cuda/FIRFilter.h TU/cuda/Array++.h TU/cuda/allocator.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/range.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/algorithm.h TU/cuda/algorithm.h
FIRGaussianConvolver.o: TU/cuda/FIRGaussianConvolver.h TU/cuda/FIRFilter.h \
	TU/cuda/Array++.h TU/cuda/allocator.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/range.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/algorithm.h \
	TU/cuda/algorithm.h
chrono.o: TU/cuda/chrono.h
