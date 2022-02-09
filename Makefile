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
		/usr/local/include/TU/Camera++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/algorithm.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/pair.h \
		/usr/local/include/TU/range.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/type_traits.h
HDRS		= TU/cu/Array++.h \
		TU/cu/BoxFilter.h \
		TU/cu/FIRFilter.h \
		TU/cu/FIRGaussianConvolver.h \
		TU/cu/GuidedFilter.h \
		TU/cu/ICIA.h \
		TU/cu/StereoUtility.h \
		TU/cu/Texture.h \
		TU/cu/algorithm.h \
		TU/cu/allocator.h \
		TU/cu/chrono.h \
		TU/cu/fp16.h \
		TU/cu/functional.h \
		TU/cu/iterator.h \
		TU/cu/npp.h \
		TU/cu/tuple.h \
		TU/cu/vec.h
SRCS		= FIRFilter.cu \
		FIRGaussianConvolver.cu \
		chrono.cc
OBJS		= FIRFilter.o \
		FIRGaussianConvolver.o \
		chrono.o

#include $(PROJECT)/lib/rtc.mk		# modified: CPPFLAGS, LIBS
#include $(PROJECT)/lib/cnoid.mk	# modified: CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# added:    PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
FIRFilter.o: TU/cu/FIRFilter.h TU/cu/Array++.h TU/cu/allocator.h \
	TU/cu/iterator.h /usr/local/include/TU/range.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/type_traits.h /usr/local/include/TU/algorithm.h \
	TU/cu/tuple.h /usr/local/include/TU/Array++.h TU/cu/algorithm.h \
	TU/cu/vec.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/Camera++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Vector++.h
FIRGaussianConvolver.o: TU/cu/FIRGaussianConvolver.h TU/cu/FIRFilter.h \
	TU/cu/Array++.h TU/cu/allocator.h TU/cu/iterator.h \
	/usr/local/include/TU/range.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/type_traits.h \
	/usr/local/include/TU/algorithm.h TU/cu/tuple.h \
	/usr/local/include/TU/Array++.h TU/cu/algorithm.h TU/cu/vec.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/Manip.h /usr/local/include/TU/Camera++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/Vector++.h
chrono.o: TU/cu/chrono.h
