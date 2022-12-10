# Makefile

CC = gcc
CPPFLAGS =
CFLAGS = -Wall -Wextra -O3 `pkg-config --cflags sdl2 SDL2_image`
LDFLAGS =
LDLIBS = `pkg-config --libs sdl2 SDL2_image`



SRC = nn2.c file.c lexer.c
OBJ = ${SRC:.c=.o}
EXE = ${SRC:.c=}


.PHONY: clean

clean:
	${RM} ${OBJ}
	${RM} ${EXE}

# END