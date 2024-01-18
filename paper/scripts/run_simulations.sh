python3 run_individual.py --n-back 0 --threads 60
python3 run_individual.py --n-back 1 --threads 60
python3 run_individual.py --n-back 2 --threads 60
python3 run_pairs.py --n-back 0 --interaction-type strict --threads 60
python3 run_pairs.py --n-back 1 --interaction-type strict --threads 60
python3 run_pairs.py --n-back 2 --interaction-type strict --threads 60
python3 run_pairs.py --n-back 0 --interaction-type flexible --threads 60
python3 run_pairs.py --n-back 1 --interaction-type flexible --threads 60
python3 run_pairs.py --n-back 2 --interaction-type flexible --threads 60
python3 run_pairs.py --n-back 0 --interaction-type shortest --threads 60
python3 run_pairs.py --n-back 1 --interaction-type shortest --threads 60
python3 run_pairs.py --n-back 2 --interaction-type shortest --threads 60
