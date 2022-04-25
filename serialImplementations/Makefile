CC = g++
CFLAGS = -O0 -fopenmp
PROJECT = traditionalCannyAlgorithm
PROJECT_SERIAL = traditionalCannyAlgorithm_serial
PROJECT_SERIAL_REFERENCE = openCVCannyImplementation
SRC = traditionalCannyAlgorithm.cpp
SRC_REF = openCVCannyImplementation.cpp

LIBS = `pkg-config --cflags --libs opencv4`

all: serial parallel

$(PROJECT) : $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(PROJECT) $(LIBS)

serial:
	$(CC) $(SRC) -o $(PROJECT_SERIAL) $(LIBS)

parallel:
	$(CC) $(CFLAGS) $(SRC) -o $(PROJECT) $(LIBS)

serial_reference:
	$(CC) $(SRC_REF) -o $(PROJECT_SERIAL_REFERENCE) $(LIBS)