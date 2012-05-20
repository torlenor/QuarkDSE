#!/bin/bash
./utils/template-converter kernel_str < kernel.cl > kernel.h
g++ -O3 -o ./main_mac.x main_mac.cpp -framework OpenCL 
