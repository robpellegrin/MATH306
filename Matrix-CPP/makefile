# Created by: Robert Pellegrin 11/06/2024

# Source and header files for project.
SRC_FILES = main.cpp Matrix.cpp
HEADER_FILES = Matrix.hpp

# Name of the compiled binary will be same as working directory.
PROG_NAME = $(shell basename $(PWD))

# Compiler to use.
CC = g++ 
# Compile-time flags.
CPPFLAGS = -Wall -Wextra -Wuninitialized -fopenmp -lgmp
# Link-time flags.
LDFLAGS = -fopenmp -lgmp 

# Automatically generate object files based on the source files.
OBJS = $(SRC_FILES:.cpp=.o)

# Binary and all build-related files will be stored in ./bin/
OUTDIR = ./bin
OBJECT_FILES = $(addprefix $(OUTDIR)/, $(OBJS))

# Make bin directory if it does not exist.
MKDIR_P = mkdir -p

# .PHONY tells Make that the listed targets are not related to actual files.
.PHONY: directories all clean run rebuild val

# 'all' target depends on 'directories' target, ensuring that the necessary
# directories (such as $(OUTDIR)) are created before building the program.
all: directories $(OUTDIR)/$(PROG_NAME)

# 'directories' target depends on '$(OUTDIR)' (./bin/), ensuring the
# output directory is created before proceeding with other build steps.
directories: $(OUTDIR)

# Make OUTDIR if it does not exist.
$(OUTDIR):
	${MKDIR_P} ${OUTDIR}

# Compile each source file (.cpp) into an object file (.o) in $(OUTDIR),
# using $(CPPFLAGS) for compile-time options. $(HEADER_FILES) are included
# as dependencies, ensuring that changes to headers trigger recompilation.
$(OUTDIR)/%.o: %.cpp $(HEADER_FILES)
	$(CC) $(CPPFLAGS) -c -o $@ $<

# Link object files to create the final executable in $(OUTDIR)/$(PROG_NAME)
# $(OBJECT_FILES) are the compiled object files (.o), and $(LDFLAGS) are
# the linker flags to link against necessary libraries.
$(OUTDIR)/$(PROG_NAME): $(OBJECT_FILES)
	$(CC) $(LDFLAGS) -o $@ $^

# 'make val' runs valgrind on output binary.
val:
	valgrind $(OUTDIR)/$(PROG_NAME) 2>&1 | tee $(OUTDIR)//$(PROG_NAME)-valgrind.log

# 'make run' executes the binary.
run: all
	@echo "|---- START ----|"
	@$(OUTDIR)/$(PROG_NAME)
	@echo "|----- END  ----|"

clean:
	rm -f core
	rm -rf $(OUTDIR)

rebuild: clean all
