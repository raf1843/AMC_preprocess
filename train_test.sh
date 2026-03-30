#! /bin/bash

classifier="LDA"   # choose from 'LRG' 'LDA' 'SGD' 
cuda=false   	   # true or false

#python3 gen_subset.py
#python3 gen_transfer_subset.py
#python3 linear_trans.py
if $cuda; then
	#python3 second_trans.py  --classifier $classifier  --cuda
	python3 high_trans.py  --classifier $classifier  --cuda
else
	#python3 second_trans.py  --classifier $classifier
	python3 high_trans.py  --classifier $classifier
fi
