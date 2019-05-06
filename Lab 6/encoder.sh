#!/bin/bash
if [ "$#" -eq 1 ]
then
    python encoder.py $1 1.0
elif [ "$#" -eq 2 ]
then
    python encoder.py $1 $2
else
	echo "Usage: ./encoder.sh gridfile"
	echo "Usage: ./encoder.sh gridfile probabilty"
fi
