compile_cuda_debug: 
	nvcc -g -G -o ./artifacts/currentCudaExecutable ${FILE}
compile_cuda: 
	nvcc -o ./artifacts/currentCudaExecutable ${FILE}