lib.name = nn_audio

class.sources = src/linreg~.c src/updown~.c src/updown2~.c src/down~.c src/nn3~.c src/nn4~.c src/nn5~.c src/nnpulse~.c

# CFLAGS = -mavx -DUSE_FLOAT

PDLIBBUILDER_DIR=pd-lib-builder/
include ${PDLIBBUILDER_DIR}/Makefile.pdlibbuilder
