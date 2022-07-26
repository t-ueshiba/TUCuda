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

INCDIRS		= -I../.. -I$(PREFIX)/include -I$(CUDAHOME)/include
CPPFLAGS	= -DNDEBUG -DUSE_TBB
CFLAGS		= -O3
NVCCFLAGS	= -O3 #-arch=sm_53 -Xcompiler -mf16c
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
		../../TU/cu/Morphology.h \
		../../TU/cu/algorithm.h \
		../../TU/cu/allocator.h \
		../../TU/cu/chrono.h \
		../../TU/cu/functional.h \
		../../TU/cu/iterator.h \
		../../TU/cu/tuple.h \
		../../TU/cu/vec.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Camera++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Profiler.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/algorithm.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/pair.h \
		/usr/local/include/TU/range.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/type_traits.h
HDRS		=
SRCS		= main.cu
OBJS		= main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
main.o: /usr/local/include/TU/Image++.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/type_traits.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/Camera++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/range.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/algorithm.h /usr/local/include/TU/Profiler.h \
	../../TU/cu/Array++.h ../../TU/cu/allocator.h ../../TU/cu/iterator.h \
	../../TU/cu/tuple.h ../../TU/cu/algorithm.h ../../TU/cu/functional.h \
	../../TU/cu/chrono.h ../../TU/cu/Morphology.h ../../TU/cu/vec.h
