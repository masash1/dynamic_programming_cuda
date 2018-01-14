test: test.cu
	nvcc -arch=sm_52 -rdc=true test.cu -lcudadevrt
