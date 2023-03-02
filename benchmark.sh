#!/bin/bash

source venv/bin/activate;

for i in {1..200}; do
  python vardec/benchmark_gen_add.py "$i";
done

for i in {1..200}; do
  python vardec/benchmark_gen_spaces.py "$i";
done

deactivate;