#!/bin/bash

source venv/bin/activate;

python vardec/benchmark.py --name spaces --fast;
python vardec/benchmark.py --name add --mondec;
python vardec/benchmark.py --name add;
python vardec/benchmark.py --name grid;

deactivate;