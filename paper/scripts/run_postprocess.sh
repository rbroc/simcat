#!/bin/bash

python3 postprocess.py --interaction-type strict --n-back 0 --threads 60
python3 postprocess.py --interaction-type flexible --n-back 0 --threads 60 
python3 postprocess.py --interaction-type shortest --n-back 0 --threads 60
python3 postprocess.py --interaction-type strict --n-back 1 --threads 60
python3 postprocess.py --interaction-type flexible --n-back 1 --threads 60
python3 postprocess.py --interaction-type shortest --n-back 1 --threads 60
python3 postprocess.py --interaction-type strict --n-back 2 --threads 60
python3 postprocess.py --interaction-type flexible --n-back 2 --threads 60
python3 postprocess.py --interaction-type shortest --n-back 2 --threads 60
cd ../metrics/0_back
gzip aggregates_strict.tsv -f
gzip aggregates_flexible.tsv -f
gzip aggregates_shortest.tsv -f
cd ../1_back
gzip aggregates_strict.tsv -f 
gzip aggregates_flexible.tsv -f
gzip aggregates_shortest.tsv -f
cd ../2_back
gzip aggregates_strict.tsv -f 
gzip aggregates_flexible.tsv -f 
gzip aggregates_shortest.tsv -f
cd ../../../../scripts
