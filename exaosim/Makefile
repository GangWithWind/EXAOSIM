all: wfscu

testccd:
	nvcc -ccbin gcc -I/usr/local/cuda-10.1/samples/common/inc -m64 -o build/ccd ./src/ccdlib.cu src/array.cu -lm -lcufft

wfscu:
	nvcc -ccbin gcc  -I/usr/local/cuda-10.1/samples/common/inc  -m64 -o ./build/wfs_g src/array.cu src/wfs.cu -lm -lcufft

cuwfslib2:
	nvcc -o lib/libcuwfs2.so -shared -Xcompiler -fPIC src/wfslib2.cu src/array.cu -m64 -lm -lcufft

cuwfslib:
	nvcc -o lib/libcuwfs.so -shared -Xcompiler -fPIC src/wfslib.cu src/array.cu -m64 -lm -lcufft

testcu:
	nvcc -o lib/libtestcu.so -shared -Xcompiler -fPIC src/foo.cu
	gcc src/test.c -Wl,-rpath,./lib -L./lib -ltestcu  -o lib/cu2
	# nvcc -ccbin gcc -m64 -Xcompiler -fPIC -o lib/cu.o -c src/foo.cu
	# gcc -shared -fPIC -o lib/cu.o lib/libcu.so
	# gcc lib/libcu.so src/test.c -o lib/test

testc:
	gcc -o lib/libpycall.so -shared -fPIC src/foo.c

ao_c: 
	gcc -o ./build/ao_c c/ao.c lib/array.c -lm

ao_g:
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o ./build/ao_g lib/array.cu cuda/ao.cu -lm

bunch.o:bunch.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bunch: bunch.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	./lib/cu
	./lib/cu

clean:
	rm -f ./build/ao_c
	rm -f ./build/ao_g