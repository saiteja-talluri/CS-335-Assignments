#!/bin/bash
if [ "$#" -eq 2 ]
then
    python decoder.py $1 $2 1.0
elif [ "$#" -eq 3 ]
then
    python decoder.py $1 $2 $3
else
	echo "Usage: ./decoder.sh gridfile value_and_policy_file"
	echo "Usage: ./decoder.sh gridfile value_and_policy_file probabilty"
fi