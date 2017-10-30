SUBDIR	= boxFilterTest		\
	  extremaTest		\
	  filterTest		\
	  gaussTest		\
	  mapTest		\
	  op3x3Test		\
	  subsampleTest		\
	  testIIDCcuda		\
	  transposeTest

TARGETS	= all clean depend

all:

$(TARGETS):
	@for d in $(SUBDIR); do				\
	  echo "";					\
	  echo "*** Current directory: $$d ***";	\
	  cd $$d;					\
	  $(MAKE) NAME=$$d $@;				\
	  cd ..;					\
	done
