# comment out to disable OpenCL support
OPENCL_SUPPORT = 1

DEBUG = 1

MAJOR = 0
MINOR = 3
NAME = hybridmc
VERSION = $(MAJOR).$(MINOR)

TARGET_LIB_LINKER_NAME = lib${NAME}.so
TARGET_LIB_SONAME = lib$(NAME).so.$(MAJOR)
TARGET_LIB_REAL_NAME = lib$(NAME).so.$(VERSION)

INSTALL_DIR=/usr/local
FENICS_INSTALL_DIR=/home/kapamaroo/Work/FEniCS

SRC_DIR=mc
INCLUDE_DIR=${SRC_DIR}/include
BUILD_DIR=build
SWIG_DIR=${BUILD_DIR}/swig
PYTHON_MODULES_DIR=site-packages/${NAME}

# OPENCL_PATH=/opt/intel/opencl-1.2-3.0.67279/lib64/
OPENCL_PATH=/usr/local/cuda-6.5/targets/x86_64-linux/lib

TEST_DIR=test
CPP_MC_TEST_FILE = ${TEST_DIR}/cpp/mc-opencl.cpp
PYTHON_MC_TEST_FILE = ${TEST_DIR}/python/mc2D.py

CXX = g++
RM = rm -rf

ifdef DEBUG
CXXFLAGS = -g -UNDEBUG
else
CXXFLAGS = -O3 -march=native -DNDEBUG
endif

OPENCL_FLAGS = -L${OPENCL_PATH} -lOpenCL

CXXFLAGS += -fPIC -std=c++11 `pkg-config --cflags gsl` -lrt
CXXFLAGS += -Wall -MP -MMD -I./${INCLUDE_DIR} -I/usr/include/lam -I$(FENICS_INSTALL_DIR)/include
LDFLAGS = -shared -Wl,-soname,$(TARGET_LIB_SONAME) -lpthread -lm `pkg-config --libs gsl`

ifdef OPENCL_SUPPORT
API = -DOPENCL_SUPPORT
CXXFLAGS += $(API)
#LDFLAGS += ${OPENCL_FLAGS}
endif

TEST_FLAGS = -I$(FENICS_INSTALL_DIR)/include
#TEST_FLAGS += -I./${INCLUDE_DIR}

DEPS = $(wildcard ${INCLUDE_DIR}/*.h)
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o, $(SRCS))

DSO_CC=-fPIC

MPI_EXECUTABLE = "hmc_mpi_exec"

-include $(SRCS:.cpp=.d)

.PHONY: all
all: _pre lib$(NAME) python-lib $(MPI_EXECUTABLE)
#test

.PHONY: lib$(NAME)
lib$(NAME): lib/$(TARGET_LIB_LINKER_NAME)

.PHONY: _pre
_pre:
	mkdir -p $(BUILD_DIR)
	mkdir -p lib

ifdef OPENCL_SUPPORT
	@echo > $(INCLUDE_DIR)/kernel_paths.h
	@echo "#define MC_KERNEL_2D \"$(INSTALL_DIR)/share/$(NAME)/MonteCarlo2D.cl\"" >> $(INCLUDE_DIR)/kernel_paths.h
	@echo "#define MC_KERNEL_3D \"$(INSTALL_DIR)/share/$(NAME)/MonteCarlo3D.cl\"" >> $(INCLUDE_DIR)/kernel_paths.h
	@echo >> $(INCLUDE_DIR)/kernel_paths.h
endif

	@echo >$(INCLUDE_DIR)/mpi_exec_path.h
	@echo "#define MPI_EXEC_BINARY \"$(INSTALL_DIR)/bin/$(MPI_EXECUTABLE)\"" >>$(INCLUDE_DIR)/mpi_exec_path.h
	@echo >>$(INCLUDE_DIR)/mpi_exec_path.h

.PHONY: debug-test
debug-test: test
	LD_LIBRARY_PATH=lib:$(LD_LIBRARY_PATH) gdb --ex r --args $(BUILD_DIR)/test $(MC_TEST_ARGS)

.PHONY: run-test
run-test: test
	LD_LIBRARY_PATH=lib:$(LD_LIBRARY_PATH) $(BUILD_DIR)/test $(MC_TEST_ARGS)

$(BUILD_DIR)/test.o: ${CPP_MC_TEST_FILE}
	$(CXX) -std=c++11 -g -Wall $(TEST_FLAGS) -c -o $@ $(CPP_MC_TEST_FILE)

$(BUILD_DIR)/test: $(BUILD_DIR)/test.o lib$(NAME)
	$(CXX) -Llib -Wall $(BUILD_DIR)/test.o -l$(NAME) -o $@

.PHONY: test
test: $(BUILD_DIR)/test

lib/$(TARGET_LIB_REAL_NAME): $(OBJS)
	$(CXX) $^ $(LDFLAGS) ${OPENCL_FLAGS} -o $@

lib/$(TARGET_LIB_SONAME): lib/$(TARGET_LIB_REAL_NAME)
	ln -sf $(TARGET_LIB_REAL_NAME) lib/$(TARGET_LIB_SONAME)

lib/$(TARGET_LIB_LINKER_NAME): lib/$(TARGET_LIB_SONAME)
#	ldconfig -v -n lib/
	ln -sf $(TARGET_LIB_SONAME) lib/$(TARGET_LIB_LINKER_NAME)

PHONY: $(MPI_EXECUTABLE)
$(MPI_EXECUTABLE):
	mpic++ $(SRC_DIR)/mpi_impl.c -o $(BUILD_DIR)/$(MPI_EXECUTABLE) $(CXXFLAGS) $(OPENCL_FLAGS) `pkg-config --libs gsl`

install:
# install headers
	mkdir -p $(INSTALL_DIR)/include/$(NAME)
	cp -r $(INCLUDE_DIR)/main.h $(INSTALL_DIR)/include/$(NAME)/mc.h

# install C++ library
	cp -d lib/* $(INSTALL_DIR)/lib

# prepare Python module
	cp site-packages/src/* ${PYTHON_MODULES_DIR}
	mv ${PYTHON_MODULES_DIR}/hmc_api.py ${PYTHON_MODULES_DIR}/${NAME}.py

# install Python module
#cp -r site-packages/$(NAME) $(INSTALL_DIR)/lib/python2.7/site-packages
	cd site-packages;\
	python setup.py install --prefix=$(INSTALL_DIR) --record python_install_files.log;\
	cd ..;

ifdef OPENCL_SUPPORT
# install shared files, kernels, etc
	mkdir -p $(INSTALL_DIR)/share/$(NAME)
	cp $(SRC_DIR)/MonteCarlo?D.cl $(INSTALL_DIR)/share/$(NAME)
endif

	cp $(BUILD_DIR)/$(MPI_EXECUTABLE) $(INSTALL_DIR)/bin

$(BUILD_DIR)/%.o: $(SRCS) $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

uninstall:
# remove headers
	rm -rf $(INSTALL_DIR)/include/$(NAME)

# remove C++ library
	rm -f $(INSTALL_DIR)/lib/$(TARGET_LIB_REAL_NAME)
	rm -f $(INSTALL_DIR)/lib/$(TARGET_LIB_SONAME)
	rm -f $(INSTALL_DIR)/lib/$(TARGET_LIB_LINKER_NAME)

# remove Python module
#rm -rf $(INSTALL_DIR)/lib/python2.7/site-packages/$(NAME)
	cat site-packages/python_install_files.log | xargs rm -rf
	rm site-packages/python_install_files.log

ifdef OPENCL_SUPPORT
# remove shared files, kernels, etc
	 rm -rf $(INSTALL_DIR)/share/$(NAME)
endif

.PHONY: clean
clean:
#	make -C ${SRC_DIR} clean
	-${RM} $(SRC_DIR)/*~ core $(BUILD_DIR) site-packages/build lib experiment $(NAME).egg-info site-packages/build site-packages/dist site-packages/$(NAME).egg-info $(INCLUDE_DIR)/kernel_paths.h $(INCLUDE_DIR)/mpi_exec_path.h ${PYTHON_MODULES_DIR}

################################################
# SWIG: generate python interface

.PHONY: wrappers
wrappers: swig_build_dirs
	swig $(API) -I${INCLUDE_DIR} -Wall -module _${NAME} -python -c++ -o ${SWIG_DIR}/wrappers/main_wrap.cxx -outdir ${PYTHON_MODULES_DIR} swig/main.i

swig_build_dirs:
	-${RM} $(SWIG_DIR) $(PYTHON_MODULES_DIR)

	mkdir -p $(SWIG_DIR)
	mkdir -p $(SWIG_DIR)/obj
	mkdir -p $(SWIG_DIR)/wrappers
	mkdir -p $(PYTHON_MODULES_DIR)

.PHONY: run-python-test
run-python-test:
	LD_LIBRARY_PATH=lib:$(LD_LIBRARY_PATH) PYTHONPATH=${PYTHON_MODULES_DIR}:${PYTHONPATH} python ${PYTHON_MC_TEST_FILE} ${MC_TEST_ARGS}

.PHONY: debug-python
debug-python:
	LD_LIBRARY_PATH=lib:$(LD_LIBRARY_PATH) PYTHONPATH=${PYTHON_MODULES_DIR}:${PYTHONPATH} gdb --ex r --args python ${PYTHON_MC_TEST_FILE} ${MC_TEST_ARGS}

.PHONY: python-lib
python-lib: wrappers
	$(CXX) -c ${SWIG_DIR}/wrappers/main_wrap.cxx $(CXXFLAGS) -o ${SWIG_DIR}/obj/main_wrap.o `pkg-config --cflags python` -I${INCLUDE_DIR}
	$(CXX) -shared -Llib -lhybridmc ${BUILD_DIR}/main.o ${SWIG_DIR}/obj/main_wrap.o ${OPENCL_FLAGS} `pkg-config --libs gsl` -o ${PYTHON_MODULES_DIR}/__${NAME}.so

# build module
#	cd site-packages;\
#	python setup.py build; \
#	cd ..;

################################################

archive:
	git archive --format=zip master -o lib$(NAME)-$(VERSION).zip
