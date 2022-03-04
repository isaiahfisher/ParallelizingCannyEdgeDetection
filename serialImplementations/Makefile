CC = g++
PROJECT = traditionalCannyAlgorithm
SRC = traditionalCannyAlgorithm.cpp

LIBS = `pkg-config --cflags --libs opencv4`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)
