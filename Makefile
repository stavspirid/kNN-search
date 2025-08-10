CC = gcc
CFLAGS = -g -Wall -O2
LIBS = -lopenblas -lm

TARGET = knn-single-thread
SOURCES = knn.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LIBS) -o $(TARGET)

clean:
	-rm -f *.o
	-rm -f $(TARGET)

.PHONY: all clean
