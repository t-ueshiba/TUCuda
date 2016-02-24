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

INCDIRS		= -I. -I$(HOME)/src/TUTools++ -I$(PREFIX)/include -I$(CUDAHOME)/include
CPPFLAGS	= -DNDEBUG #-DSSE3
CFLAGS		= -O
NVCCFLAGS	= -O
CCFLAGS		= $(CFLAGS)

LIBS		=
ifneq ($(findstring darwin,$(OSTYPE)),)
  LIBS	       += -framework IOKit -framework CoreFoundation -framework CoreServices
endif

LINKER		= $(NVCC)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= /home/ueshiba/src/TUTools++/TU/Array++.h \
		/home/ueshiba/src/TUTools++/TU/functional.h \
		/home/ueshiba/src/TUTools++/TU/iterator.h \
		/home/ueshiba/src/TUTools++/TU/pair.h \
		/home/ueshiba/src/TUTools++/TU/tuple.h
HDRS		= TU/cuda/Array++.h \
		TU/cuda/FIRFilter.h \
		TU/cuda/FIRGaussianConvolver.h \
		TU/cuda/Texture.h \
		TU/cuda/utility.h
SRCS		= FIRFilter.cu \
		FIRGaussianConvolver.cc \
		op3x3.cu \
		subsample.cu \
		suppressNonExtrema3x3.cu
OBJS		= FIRFilter.o \
		FIRGaussianConvolver.o \
		op3x3.o \
		subsample.o \
		suppressNonExtrema3x3.o

#include $(PROJECT)/lib/rtc.mk		# modified: CPPFLAGS, LIBS
#include $(PROJECT)/lib/cnoid.mk	# modified: CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# added:    PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
FIRFilter.o: TU/cuda/FIRFilter.h TU/cuda/Array++.h \
	/home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/functional.h \
	/home/ueshiba/src/TUTools++/TU/pair.h TU/cuda/utility.h
FIRGaussianConvolver.o: TU/cuda/FIRGaussianConvolver.h TU/cuda/FIRFilter.h \
	TU/cuda/Array++.h /home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/functional.h \
	/home/ueshiba/src/TUTools++/TU/pair.h
op3x3.o: TU/cuda/utility.h TU/cuda/Array++.h \
	/home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/functional.h \
	/home/ueshiba/src/TUTools++/TU/pair.h
subsample.o: TU/cuda/utility.h TU/cuda/Array++.h \
	/home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/functional.h \
	/home/ueshiba/src/TUTools++/TU/pair.h
suppressNonExtrema3x3.o: TU/cuda/utility.h TU/cuda/Array++.h \
	/home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/functional.h \
	/home/ueshiba/src/TUTools++/TU/pair.h
