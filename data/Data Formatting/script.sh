#!/bin/bash
FILES=data3/*
for f in $FILES
do
  log_file="$f"
  filename=$f

  label=${f#data/}
  label=${label%.csv}
  echo $f
  java-introcs -Xmx256m csvToMatrixExtrap < $f > matricies/${label}mat.txt 

done


