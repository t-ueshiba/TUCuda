#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
PROGRAM		= $(shell basename $(PWD))
#LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I../.. -I$(HOME)/src/TUTools++ -I$(PREFIX)/include -I$(CUDAHOME)/include
CPPFLAGS	= -DNDEBUG
CFLAGS		= -O3
NVCCFLAGS	= -O3
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUTools++ -lTUCuda++ -ltbb -lnppif

LINKER		= $(NVCC)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= ../../TU/cuda/Array++.h \
		../../TU/cuda/algorithm.h \
		../../TU/cuda/allocator.h \
		../../TU/cuda/chrono.h \
		../../TU/cuda/tuple.h \
		/home/ueshiba/src/TUTools++/TU/Array++.h \
		/home/ueshiba/src/TUTools++/TU/Camera++.h \
		/home/ueshiba/src/TUTools++/TU/Geometry++.h \
		/home/ueshiba/src/TUTools++/TU/Image++.h \
		/home/ueshiba/src/TUTools++/TU/Manip.h \
		/home/ueshiba/src/TUTools++/TU/Minimize.h \
		/home/ueshiba/src/TUTools++/TU/Profiler.h \
		/home/ueshiba/src/TUTools++/TU/Vector++.h \
		/home/ueshiba/src/TUTools++/TU/algorithm.h \
		/home/ueshiba/src/TUTools++/TU/iterator.h \
		/home/ueshiba/src/TUTools++/TU/pair.h \
		/home/ueshiba/src/TUTools++/TU/range.h \
		/home/ueshiba/src/TUTools++/TU/tuple.h \
		/home/ueshiba/src/TUTools++/TU/type_traits.h
HDRS		=
SRCS		= main.cu
OBJS		= main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
main.o: /home/ueshiba/src/TUTools++/TU/Image++.h \
	/home/ueshiba/src/TUTools++/TU/pair.h \
	/home/ueshiba/src/TUTools++/TU/type_traits.h \
	/home/ueshiba/src/TUTools++/TU/Manip.h \
	/home/ueshiba/src/TUTools++/TU/Camera++.h \
	/home/ueshiba/src/TUTools++/TU/Geometry++.h \
	/home/ueshiba/src/TUTools++/TU/Minimize.h \
	/home/ueshiba/src/TUTools++/TU/Vector++.h \
	/home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/range.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/algorithm.h ../../TU/cuda/tuple.h \
	/home/ueshiba/src/TUTools++/TU/Profiler.h ../../TU/cuda/Array++.h \
	../../TU/cuda/allocator.h ../../TU/cuda/algorithm.h \
	../../TU/cuda/chrono.h
