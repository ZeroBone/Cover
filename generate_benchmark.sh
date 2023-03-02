#!/bin/bash

source venv/bin/activate;

for i in {1..12}; do
  python vardec/benchmark_gen_add.py "$i";
done

for i in {1..60}; do
  python vardec/benchmark_gen_spaces.py "$i";
done

for i in {1..64}; do
  python vardec/benchmark_gen_grid.py 2 32 "$i";
done

deactivate;