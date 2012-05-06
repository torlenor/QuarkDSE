#!/bin/bash
./utils/template-converter kernel_str < kernel.cl > kernel.h
g++ -O3 -o ./main.x main.cpp -I/opt/AMDAPP/include/ -I./include/ -lOpenCL
