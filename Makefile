CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

TARGET = kmeans_rfm
SRCS = main.c kmeans.c

all: $(TARGET)

$(TARGET): $(SRCS) kmeans.h
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.exe

.PHONY: all clean