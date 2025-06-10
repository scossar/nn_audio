lib.name = nn_audio

class.sources = src/linreg~.c  src/nn3~.c src/nn4~.c src/nn5~.c src/nnpulse~.c src/nnpulse2~.c src/nnpulse3~.c src/nnlfo~.c src/nnlfofl~.c src/lorenz~.c src/lorenz_z~.c src/lorenz_y~.c src/lorenzrk4~.c src/duffing~.c src/duffingeuler~.c src/doublependulum~.c src/rossler~.c src/lorenzattr~.c src/chua~.c src/chuaforce~.c src/henon~.c src/logistic~.c src/clogistic~.c src/tent~.c src/tentmap~.c

# removed from class.sources for now
#src/updown~.c src/updown4~.c src/down~.c

PDLIBBUILDER_DIR=pd-lib-builder/
include ${PDLIBBUILDER_DIR}/Makefile.pdlibbuilder
