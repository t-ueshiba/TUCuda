SUBDIR	= boxFilterTest		\
	  eigen33Test		\
	  extremaTest		\
	  extremaFilterTest	\
	  filterTest		\
	  fp16Test		\
	  gaussTest		\
	  guidedFilterTest	\
	  mappedMemoryTest	\
	  op3x3Test		\
	  cudaStereoTest	\
	  subsampleTest		\
	  testIIDCcuda		\
	  transposeTest		\
	  warpTest

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
