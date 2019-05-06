#!/bin/bash

if [ "$1" = "mdp" ]; then
	iter=0
	for file in data/mdp/mdpfile*.txt ; do
		((++iter))
		name=$(echo $file | cut -f1 -d-)
		sol=data/mdp/solution0"$iter".txt
		mysol=output/mdp/solution0"$iter".txt
		diff_out=output/mdp/diff0"$iter".txt
		./valueiteration.sh $name > $mysol
		diff $sol $mysol > $diff_out
	done
else
	iter=0
	for file in data/maze/grid*.txt ; do
		((++iter))
		name=data/maze/grid"$iter"0.txt
		mdpfile=output/maze/mdp/mdpfile"$iter"0.txt
		policyfile=output/maze/policy/value_and_policy_file"$iter"0.txt
		sol=data/maze/solution"$iter"0.txt
		mysol=output/maze/path/solution"$iter"0.txt
		diff_out=output/maze/diff/diff"$iter"0.txt

		if [ "$#" -eq 1 ]
		then
		    ./encoder.sh $name > output/maze/mdp/mdpfile"$iter"0.txt
		elif [ "$#" -eq 2 ]
		then
		    ./encoder.sh $name $2 > $mdpfile
		fi
		./valueiteration.sh $mdpfile > $policyfile

		if [ "$#" -eq 1 ]
		then
		    ./decoder.sh $name $policyfile > $mysol
		elif [ "$#" -eq 2 ]
		then
		    ./decoder.sh $name $policyfile $2 > $mysol
		fi
		diff $sol $mysol > $diff_out
	done
fi