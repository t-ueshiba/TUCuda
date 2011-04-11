#
#  $Id: Makefile,v 1.6 2011-04-11 08:05:54 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(CUDAHOME)/include/TU
INCDIRS		= -I. -I$(PREFIX)/include -I$(CUDASDK)/common/inc

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -g -O
NVCCFLAGS	= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O	# -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS     += -xSSE3
  else
    CPPFLAGS   += -DSSE3 
    CFLAGS     += -xSSE3
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC
EXTHDRS		=
HDRS		= CudaArray++.h \
		CudaFilter.h
SRCS		=
OBJS		=

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

include $(PROJECT)/lib/l.mk
###
