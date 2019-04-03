#! /bin/bash

for f in $@
do
	if [ $# == 2 ]; then
		echo $f
	fi
	grep "Detected" $f | cut -d " " -f 7 | cut -d "." -f 1 
done
