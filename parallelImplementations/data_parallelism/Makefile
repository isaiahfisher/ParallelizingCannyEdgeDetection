CC_SCHOLAR = icc
CFLAGS_SCHOLAR_PARALLEL = -std=c++11 -qopenmp
PROJECT_DATA_PARALLEL = cannyOpenMPDataParallel
SRC = cannyOpenMPDataParallel.cpp
LIBS = `pkg-config --cflags --libs opencv`

all: data_parallel

data_parallel:
	$(CC_SCHOLAR) $(CFLAGS_SCHOLAR_PARALLEL) $(SRC) -o $(PROJECT_DATA_PARALLEL) $(LIBS)
