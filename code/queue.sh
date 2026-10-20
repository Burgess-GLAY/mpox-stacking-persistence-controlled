for s in 42 43 44 45 46; do python3 run10.py primary v10 $s; done
python3 run10.py loco v10 42
for f in v9 env_only history_only; do python3 run10.py primary $f 42; done
python3 run10.py loco v9 42
