#!/usr/bin/env bash
# Usage: nvcc_build.sh <name> <nvcc> <flags...> <src> -o <out> <libs...>
# Compiles, keeps the full ptxas log in build/ptxas_<name>.log, and prints one
# line per kernel: registers, spill-store bytes, mangled kernel name.
set -u
name=$1; shift
mkdir -p build results
"$@" 2> "build/ptxas_${name}.log"
status=$?
if [ $status -ne 0 ]; then cat "build/ptxas_${name}.log"; exit $status; fi
awk '/Compiling entry function/{f=$(NF-2); gsub(/[^A-Za-z0-9_]/,"",f)}
     /spill stores/{s=$5}
     /Used [0-9]+ registers/{for(i=1;i<=NF;i++) if($i=="registers,") r=$(i-1);
       printf "  %4s regs  %4s B spill  %s\n", r, s, f}' "build/ptxas_${name}.log"
