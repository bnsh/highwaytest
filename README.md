This is my attempt to recreate the results in http://arxiv.org/pdf/1505.00387v2.pdf

Basically just run
	Usage: th main.lua [options] 
	Testing the HighwayLayers

	Options
	  -mathematica Mathematica output file [/dev/null]
	  -type        layer type: should be vanilla or highway [vanilla]
	  -set         input set: should be mnist or xor [mnist]
	  -layers      number of layers [2]
	  -size        hidden layer size [71]
	  -max_epochs  number of full passes through the training data [200]
	  -seed        torch manual random number generator seed [-1]
	  -gpuid       which gpu to use. -1 = use CPU [-1]
	  -cudnn       Use CUDNN [0]

So, I plan to do this:
	for layers in 10 20 50 100
	do
		th main.lua -gpuid 0 -cudnn 1 -mathematica "/tmp/highwaytest/"vanilla-${layers}.m" -type vanilla -set mnist -layers "${layers}" -size 71 -max_epochs 400 
		th main.lua -gpuid 1 -cudnn 1 -mathematica "/tmp/highwaytest/"highway-${layers}.m" -type highway -set mnist -layers "${layers}" -size 50 -max_epochs 400 
	done

(Tho, obviously, I'll run these in parallel on multiple shells.)
