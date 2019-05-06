#!/bin/bash

if [ "$1" = "mdp" ]; then
	iter=$2
	name=data/mdp/mdpfile0"$iter".txt
	sol=data/mdp/solution0"$iter".txt
	mysol=output/mdp/solution0"$iter".txt
	diff_out=output/mdp/diff0"$iter".txt
	./valueiteration.sh $name > $mysol
	diff $sol $mysol > $diff_out
elif [ "$1" = "maze" ]; then
	iter=$2
	name=data/maze/grid"$iter".txt
	mdpfile=output/maze/mdp/mdpfile"$iter".txt
	policyfile=output/maze/policy/value_and_policy_file"$iter".txt
	sol=data/maze/solution"$iter".txt
	mysol=output/maze/path/solution"$iter".txt
	diff_out=output/maze/diff/diff"$iter".txt

	if [ "$#" -eq 2 ]
	then
	    ./encoder.sh $name > $mdpfile
	elif [ "$#" -eq 3 ]
	then
	    ./encoder.sh $name $3 > $mdpfile
	fi
	./valueiteration.sh $mdpfile > $policyfile

	if [ "$#" -eq 2 ]
	then
	    ./decoder.sh $name $policyfile > $mysol
	elif [ "$#" -eq 3 ]
	then
	    ./decoder.sh $name $policyfile $3 > $mysol
	fi
	diff $sol $mysol > $diff_out
	python3 visualize.py $name $sol
	python3 visualize.py $name $mysol

elif [ "$1" = "test" ]; then
	iter=$2
	name=data/maze/grid"$iter".txt
	if [ "$#" -eq 2 ]
	then
	    ./encoder.sh $name > test/mdpfile"$iter"_1.0.txt
	    ./valueiteration.sh test/mdpfile"$iter"_1.0.txt > test/value_and_policy_file"$iter"_1.0.txt
	elif [ "$#" -eq 3 ]
	then
	    ./encoder.sh $name "$3" > test/mdpfile"$iter"_"$3".txt
	    ./valueiteration.sh test/mdpfile"$iter"_"$3".txt > test/value_and_policy_file"$iter"_"$3".txt
	fi
	
	if [ "$#" -eq 2 ]
	then
		echo "1.0" > test/solution"$iter"_1.0.txt
	    ./decoder.sh $name test/value_and_policy_file"$iter"_1.0.txt >> test/solution"$iter"_1.0.txt
	elif [ "$#" -eq 3 ]
	then
		echo "$3" > test/solution"$iter"_"$3".txt
	    ./decoder.sh $name test/value_and_policy_file"$iter"_"$3".txt "$3" >> test/solution"$iter"_"$3".txt
	fi
	rm test/mdpfile*
	rm test/value*
fi