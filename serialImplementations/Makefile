CC = g++
CFLAGS = -O0 -fopenmp
PROJECT = traditionalCannyAlgorithm
PROJECT_SERIAL = traditionalCannyAlgorithm_serial
SRC = traditionalCannyAlgorithm.cpp

LIBS = `pkg-config --cflags --libs opencv4`

all: serial parallel

$(PROJECT) : $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(PROJECT) $(LIBS)

serial:
	$(CC) $(SRC) -o $(PROJECT_SERIAL) $(LIBS)

parallel:
	$(CC) $(CFLAGS) $(SRC) -o $(PROJECT) $(LIBS)