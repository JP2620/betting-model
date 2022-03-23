#!/usr/bin/bash

for i in `seq 2006 1 2021`
do
  python3.8 premier.py $i $(($i+1)) >> resultados.txt &

done
