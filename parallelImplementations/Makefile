CC = g++
CC_SCHOLAR = icc
CFLAGS = -O0 -fopenmp
CFLAGS_SCHOLAR = -std=c++11
CFLAGS_SCHOLAR_PARALLEL = -std=c++11 -qopenmp
PROJECT = traditionalCannyAlgorithm
PROJECT_SERIAL = traditionalCannyAlgorithm_serial
PROJECT_SERIAL_REFERENCE = openCVCannyImplementation
PROJECT_OPENMP = cannyAlgorithmOpenMP
SRC = traditionalCannyAlgorithm.cpp
SRC_REF = openCVCannyImplementation.cpp
SRC_OPENMP = cannyAlgorithmOpenMP.cpp

LIBS = `pkg-config --cflags --libs opencv`

all: serial parallel

$(PROJECT) : $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(PROJECT) $(LIBS)

serial:
	$(CC) $(SRC) -o $(PROJECT_SERIAL) $(LIBS)

serial_scholar:
	$(CC_SCHOLAR) $(CFLAGS_SCHOLAR) $(SRC) -o $(PROJECT_SERIAL) $(LIBS)

parallel:
	$(CC) $(CFLAGS) $(SRC) -o $(PROJECT) $(LIBS)

serial_reference:
	$(CC) $(SRC_REF) -o $(PROJECT_SERIAL_REFERENCE) $(LIBS)

openmp_scholar:
	$(CC_SCHOLAR) $(CFLAGS_SCHOLAR_PARALLEL) $(SRC_OPENMP) -o $(PROJECT_OPENMP) $(LIBS)