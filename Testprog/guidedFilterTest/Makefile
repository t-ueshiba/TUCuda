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

INCDIRS		= -I../.. -I$(HOME)/src/TUTools -I$(CUDAHOME)/include
CPPFLAGS	= #-DNDEBUG -DUSE_TBB
CFLAGS		= -O3
NVCCFLAGS	= -O3 --extended-lambda #-arch=sm_53 -Xcompiler -mf16c
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUCuda -lTUTools -ltbb
LINKER		= $(NVCC)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= ../../TU/cu/Array++.h \
		../../TU/cu/BoxFilter.h \
		../../TU/cu/GuidedFilter.h \
		../../TU/cu/algorithm.h \
		../../TU/cu/allocator.h \
		../../TU/cu/chrono.h \
		../../TU/cu/iterator.h \
		../../TU/cu/tuple.h \
		../../TU/cu/vec.h \
		/home/artuser/src/TUTools/TU/Array++.h \
		/home/artuser/src/TUTools/TU/BoxFilter.h \
		/home/artuser/src/TUTools/TU/Camera++.h \
		/home/artuser/src/TUTools/TU/Filter2.h \
		/home/artuser/src/TUTools/TU/Geometry++.h \
		/home/artuser/src/TUTools/TU/GuidedFilter.h \
		/home/artuser/src/TUTools/TU/Image++.h \
		/home/artuser/src/TUTools/TU/Manip.h \
		/home/artuser/src/TUTools/TU/Minimize.h \
		/home/artuser/src/TUTools/TU/Profiler.h \
		/home/artuser/src/TUTools/TU/Vector++.h \
		/home/artuser/src/TUTools/TU/algorithm.h \
		/home/artuser/src/TUTools/TU/iterator.h \
		/home/artuser/src/TUTools/TU/pair.h \
		/home/artuser/src/TUTools/TU/range.h \
		/home/artuser/src/TUTools/TU/tuple.h \
		/home/artuser/src/TUTools/TU/type_traits.h
HDRS		=
SRCS		= cpuJob.cc \
		cudaJob.cu \
		main.cu
OBJS		= cpuJob.o \
		cudaJob.o \
		main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
cpuJob.o: /home/artuser/src/TUTools/TU/GuidedFilter.h \
	/home/artuser/src/TUTools/TU/BoxFilter.h \
	/home/artuser/src/TUTools/TU/Filter2.h \
	/home/artuser/src/TUTools/TU/iterator.h \
	/home/artuser/src/TUTools/TU/tuple.h \
	/home/artuser/src/TUTools/TU/type_traits.h \
	/home/artuser/src/TUTools/TU/Array++.h \
	/home/artuser/src/TUTools/TU/range.h \
	/home/artuser/src/TUTools/TU/algorithm.h \
	/home/artuser/src/TUTools/TU/Profiler.h
cudaJob.o: /home/artuser/src/TUTools/TU/Profiler.h ../../TU/cu/chrono.h \
	../../TU/cu/GuidedFilter.h ../../TU/cu/tuple.h \
	/home/artuser/src/TUTools/TU/tuple.h \
	/home/artuser/src/TUTools/TU/type_traits.h ../../TU/cu/vec.h \
	/home/artuser/src/TUTools/TU/Image++.h \
	/home/artuser/src/TUTools/TU/pair.h \
	/home/artuser/src/TUTools/TU/Manip.h \
	/home/artuser/src/TUTools/TU/Camera++.h \
	/home/artuser/src/TUTools/TU/Geometry++.h \
	/home/artuser/src/TUTools/TU/Minimize.h \
	/home/artuser/src/TUTools/TU/Vector++.h \
	/home/artuser/src/TUTools/TU/Array++.h \
	/home/artuser/src/TUTools/TU/range.h \
	/home/artuser/src/TUTools/TU/iterator.h \
	/home/artuser/src/TUTools/TU/algorithm.h ../../TU/cu/BoxFilter.h \
	../../TU/cu/Array++.h ../../TU/cu/allocator.h ../../TU/cu/iterator.h \
	../../TU/cu/algorithm.h
main.o: /home/artuser/src/TUTools/TU/Image++.h \
	/home/artuser/src/TUTools/TU/pair.h \
	/home/artuser/src/TUTools/TU/type_traits.h \
	/home/artuser/src/TUTools/TU/Manip.h \
	/home/artuser/src/TUTools/TU/Camera++.h \
	/home/artuser/src/TUTools/TU/Geometry++.h \
	/home/artuser/src/TUTools/TU/Minimize.h \
	/home/artuser/src/TUTools/TU/Vector++.h \
	/home/artuser/src/TUTools/TU/Array++.h \
	/home/artuser/src/TUTools/TU/range.h \
	/home/artuser/src/TUTools/TU/iterator.h \
	/home/artuser/src/TUTools/TU/tuple.h \
	/home/artuser/src/TUTools/TU/algorithm.h
