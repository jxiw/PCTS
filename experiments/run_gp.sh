#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/junxiong/Documents/HOO/MFTreeSearchCV-master/

python -u experiment_synthetic_gp.py --function=Hartmann3 > GP/GP1_Hartmann3.txt
python -u experiment_synthetic_gp.py --function=Hartmann6 > GP/GP1_Hartmann6.txt
python -u experiment_synthetic_gp.py --function=CurrinExp > GP/GP1_CurrinExp.txt
python -u experiment_synthetic_gp.py --function=Branin    > GP/GP1_Branin.txt
python -u experiment_synthetic_gp.py --function=Borehole  > GP/GP1_Borehole.txt
python -u experiment_synthetic_gp.py --function=Schwefel  > GP/GP_Schwefel_10.txt

