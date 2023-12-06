#!/usr/bin/bash

for i in `seq 2006 1 2021`
do
  python premier.py $i $(($i+1)) >> resultados.txt &

done
