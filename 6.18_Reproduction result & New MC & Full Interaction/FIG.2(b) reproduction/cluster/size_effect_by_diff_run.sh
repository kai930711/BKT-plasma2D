#!/bin/bash

set -e

N_VALUE="$1"

echo "========================================"
echo "Size effect by diff job starts"
echo "N_VALUE = ${N_VALUE}"
echo "Date:"
date
echo "Host:"
hostname
echo "PWD:"
pwd
echo "Files:"
ls -lh
echo "========================================"

echo "Python version:"
python3 --version

echo "Install packages:"
python3 -m pip install --no-cache-dir --target ./python_packages -r requirements.txt

export PYTHONPATH="$PWD/python_packages"

echo "Check imports:"
python3 -c "import numpy, pandas, scipy; print('imports OK')"

echo "Check syntax:"
python3 -m py_compile Setup.py Plasma_Interaction.py Monte_Carlo.py Size_Effect_Functions.py Size_Effect_by_Diff_OneN.py

echo "Run calculation:"
python3 -u Size_Effect_by_Diff_OneN.py "${N_VALUE}"

echo "Check output:"
ls -lh

test -f "fig2b_summary_N${N_VALUE}.csv"

echo "fig2b_summary_N${N_VALUE}.csv created successfully"

echo "Job ends:"
date
echo "========================================"
