# Makefile

CC = gcc
CPPFLAGS = -MMD
CFLAGS = -Wall -Wextra -g -std=c99 -O2
LDFLAGS = -lm
LDLIBS = -lm

SRC = nn2.c
OBJ = ${SRC:.c=.o}
DEP = ${SRC:.c=.d}

nn2: ${OBJ}

-include ${DEP}

.PHONY: clean

clean:
	${RM} ${OBJ}
	${RM} ${DEP}
	${RM} nn2

# END
