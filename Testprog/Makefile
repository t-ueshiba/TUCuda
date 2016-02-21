SUBDIR	= extremaTest		\
	  filterTest		\
	  gaussTest		\
	  op3x3Test		\
	  subsampleTest		\
	  test1394cuda

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
