#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/junxiong/Documents/HOO/MFTreeSearchCV-master/

start=1
round=10

mkdir "DHOO"
mkdir "DHOO/Hartmann3/"
mkdir "DHOO/Hartmann6/"
mkdir "DHOO/CurrinExp/"
mkdir "DHOO/Branin/"
mkdir "DHOO/Borehole/"

mkdir "HOO"
mkdir "HOO/Hartmann3/"
mkdir "HOO/Hartmann6/"
mkdir "HOO/CurrinExp/"
mkdir "HOO/Branin/"
mkdir "HOO/Borehole/"

# ucbv
for i in `seq $start $round`;
do
  # 0.02
  python -u run_dnf_synthetic.py UCBV Hartmann3 HOO 1 5 0.02 > "HOO/Hartmann3/HOO_UCBV_0.02_$i.txt"
  python -u run_dnf_synthetic.py UCBV Hartmann3 DHOO 1 5 0.02 > "DHOO/Hartmann3/DHOO_UCBV_0.02_$i.txt"
  # 0.02
  python -u run_dnf_synthetic.py UCBV Hartmann6 DHOO 1 5 0.02 > "DHOO/Hartmann6/DHOO_UCBV_0.05_$i.txt"
  python -u run_dnf_synthetic.py UCBV Hartmann6 DHOO 1 5 0.02 > "HOO/Hartmann6/HOO_UCBV_0.05_$i.txt"
  # 0.02
  python -u run_dnf_synthetic.py UCBV CurrinExp HOO 1 5 0.02 > "HOO/CurrinExp/HOO_UCBV_${j}_${i}.txt"
  python -u run_dnf_synthetic.py UCBV CurrinExp DHOO 1 5 0.02 > "DHOO/CurrinExp/DHOO_UCBV_${j}_${i}.txt"
  # 0.8
  python -u run_dnf_synthetic.py UCBV Branin HOO 1 1 0.8 > "HOO/Branin/HOO_UCBV_${j}_${i}.txt"
  python -u run_dnf_synthetic.py UCBV Branin DHOO 1 1 0.8 > "DHOO/Branin/DHOO_UCBV_${j}_${i}.txt"
  # 0.01
  python -u run_dnf_synthetic.py UCBV Borehole HOO 1 350 0.0095 > "HOO/Borehole/HOO_UCBV_${j}_${i}.txt"
  python -u run_dnf_synthetic.py UCBV Borehole DHOO 1 350 0.0095 > "DHOO/Borehole/DHOO_UCBV_${j}_${i}.txt"
done

# ucb1
for i in `seq $start $round`;
do
  python -u run_dnf_synthetic.py UCB1 Hartmann3 HOO 3 > "HOO/Hartmann3/HOO_UCB1_$i.txt"
  python -u run_dnf_synthetic.py UCB1 Hartmann3 DHOO 3 > "DHOO/Hartmann3/DHOO_UCB1_$i.txt"
  python -u run_dnf_synthetic.py UCB1 Hartmann6 HOO 3 > "HOO/Hartmann6/HOO_UCB1_$i.txt"
  python -u run_dnf_synthetic.py UCB1 Hartmann6 DHOO 3 > "DHOO/Hartmann6/DHOO_UCB1_$i.txt"
  python -u run_dnf_synthetic.py UCB1 CurrinExp HOO 3 > "HOO/CurrinExp/HOO_UCB1_$i.txt"
  python -u run_dnf_synthetic.py UCB1 CurrinExp DHOO 3 > "DHOO/CurrinExp/DHOO_UCB1_$i.txt"
  python -u run_dnf_synthetic.py UCB1 Branin HOO 3 > "HOO/Branin/HOO_UCB1_$i.txt"
  python -u run_dnf_synthetic.py UCB1 Branin DHOO 3 > "DHOO/Branin/DHOO_UCB1_$i.txt"
  python -u run_dnf_synthetic.py UCB1 Borehole HOO 3 > "HOO/Borehole/HOO_UCB1_$i.txt"
  python -u run_dnf_synthetic.py UCB1 Borehole DHOO 3 > "DHOO/Borehole/DHOO_UCB1_$i.txt"
done
