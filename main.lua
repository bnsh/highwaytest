#! /usr/local/torch/install/bin/th

require "nn"
require "nngraph"
require "optim"
require "no_globals"
local HighwayMLP = require "HighwayMLP"
local xor = require "xor"
local mnist = require "mnist"
local weight_init = require "weight-init"

local cmd = torch.CmdLine()
cmd:text('Testing the HighwayLayers')
cmd:text()
cmd:text('Options')
cmd:option('-json', '/dev/null', 'json output file')
cmd:option('-type', 'vanilla', 'layer type: should be vanilla or highway')
cmd:option('-set', 'mnist', 'input set: should be mnist or xor')
cmd:option('-layers', 2, 'number of layers')
cmd:option('-size', 71, 'hidden layer size')
cmd:option('-max_epochs', 200, 'number of full passes through the training data')
cmd:option('-seed', 12345, 'torch manual random number generator seed')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0, 'Use CUDNN')

local opt = cmd:parse(arg)
if opt.seed ~= -1 then
	torch.manualSeed(opt.seed)
end

-- load necessary packages depending on config options
if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

if opt.cudnn == 1 then
   assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
   print('using cudnn...')
   require 'cudnn'
end


local function vanilla(size, num_layers, _, f)
	local block = nn.Sequential()
	for i = 1,num_layers
	do
		block:add(nn.Linear(size, size))
		block:add(f:clone())
	end
	return block
end

local function highway(sz, num_layers, bias, f)
	return HighwayMLP.mlp(sz, num_layers, bias, f)
end

local function minibatch_generator(set, batchsz)
	local sz = set.data:size(1)
	local shuffle = torch.randperm(sz):long()
	local pos = 1
	local function generator()
		local subset = nil
		local endpos = math.min(pos + batchsz - 1,sz)
		if pos <= sz
		then
			local indices = shuffle[{{pos, endpos}}]
			local sdata = set.data:index(1, indices)
			sdata = sdata:reshape(sdata:size(1), sdata:size(2) * sdata:size(3))
			local slabel = set.label:index(1, indices):long()
			local sonehot = set.onehot:index(1, indices)
			subset = {
				data=torch.add(torch.div(sdata:double(), 255), -0.5),
				size=endpos-pos+1,
				onehot=sonehot,
				label=slabel
			}
			if opt.gpuid >= 0 then
				subset.data = subset.data:cuda()
				subset.onehot = subset.onehot:cuda()
				subset.label = subset.label:cuda()
			end
			pos = pos + batchsz
		else
			pos = 1
			shuffle = torch.randperm(sz):long()
		end
		return subset
	end
	return generator
end

local function single_epoch(trainset, crit, optimizer, mlp, mlp_parameters,mlp_gradients, optimizer_params, batchsz)
	local mbg = minibatch_generator(trainset, batchsz)
	local done = false
	local current_loss = 0

	while not done
	do
		local subset = mbg()
		done = subset == nil
		if not done
		then
			local function evaluator(x)
				assert(x == mlp_parameters)
				mlp:training()
				mlp_gradients:zero()
				local o = mlp:forward(subset.data)
				local err = crit:forward(o, subset.onehot)
				mlp:backward(subset.data, crit:backward(mlp.output, subset.onehot))
				mlp:training()
				-- mlp_gradients:clamp(-10,10)
				return err, mlp_gradients
			end
			local _, fs = optimizer(evaluator, mlp_parameters, optimizer_params)
			current_loss = current_loss + fs[1] / trainset.data:size(1)
		end
	end
	return current_loss
end

local function compute_error(mlp, crit, set, batchsz)
	mlp:evaluate()
	local s = 0
	local ss = 0
	local kl = 0
	for pos = 1,(set.data:size(1)-1),batchsz
	do
		local endpos = math.min(pos + batchsz-1, set.data:size(1))
		local rng = torch.range(pos, endpos):long()
		local input = set.data:index(1,rng):reshape(rng:size(1), set.data:size(2)*set.data:size(3)):double()
		local target = set.label:index(1,rng):long()
		local oh = set.onehot:index(1,rng)

		if opt.gpuid >= 0 then
			input = input:cuda()
			target = target:cuda()
			oh = oh:cuda()
		end

		local raw = mlp:forward(input)
		local skl = crit:forward(raw, oh)
		kl = kl + skl
		local _, output = torch.max(raw, 2)
		output = torch.add(output,-1.0)
		s = s + output:ne(target):sum()
		ss = ss + target:size(1)
	end
	assert(ss == set.data:size(1))
	mlp:training()
	return (s/set.data:size(1)), (kl/set.data:size(1))
end

local function addonehot(set)
	local identity = torch.eye(torch.max(set.label)+1)
	set.onehot = identity:index(1, torch.add(set.label:long(),1))
end

local function multiwrite(logfiles, s)
	for i,lf in ipairs(logfiles)
	do
		lf:write(s)
	end
end

local function main(argv)
	local errors = false
	local trainset = nil
	local testset = nil
	local midlayertype = nil
	if opt.set == "mnist" then
		trainset = mnist.traindataset()
		testset = mnist.testdataset()
	elseif opt.set == "xor" then
		trainset = xor.traindataset()
		testset = xor.testdataset()
	else
		errors = true
	end

	if opt.type == "vanilla" then
		midlayertype = vanilla
	elseif opt.type == "highway" then
		midlayertype = highway
	else
		errors = true
	end

	if errors then
		cmd:help()
	else
		addonehot(trainset)
		addonehot(testset)

		local layers = opt.layers
		local sz = opt.size
		local iters = opt.max_epochs
		local fh = io.open(opt.json, "w")
		local transfer = nn.ReLU()
		local optimizer = optim.adamax
		local optimizer_parameters = { }

-- This is where the neural network is constructed.
		local bias = -math.floor(layers/10)-1
		io.stderr:write(string.format("Using a bias of %.7f for a %d layer neural network\n", bias, layers))
		local mlp = nn.Sequential()
		mlp:add(nn.Linear(trainset.data:size(2)*trainset.data:size(3), sz))
		mlp:add(transfer:clone())
		mlp:add(midlayertype(sz, layers, bias, transfer))
		mlp:add(nn.Linear(sz, 1+torch.max(trainset.label)))
		mlp:add(nn.LogSoftMax())

		weight_init(mlp, "kaiming")

		local crit = nn.DistKLDivCriterion()
		crit.sizeAverage = false

		if opt.gpuid >= 0 then
			mlp:cuda()
			crit:cuda()
		end

		local mlp_parameters, mlp_gradients = mlp:getParameters()

		local logfiles = { io.stderr, io.open(opt.json, "w") }
		multiwrite(logfiles,"[")
		for iter = 0, iters
		do
			single_epoch(trainset, crit, optimizer, mlp, mlp_parameters, mlp_gradients, optimizer_parameters, 1000)
			local train_error, train_kl = compute_error(mlp, crit, trainset, 1000)
			local test_error, test_kl = compute_error(mlp, crit, testset, 1000)
			if iter > 0
			then
				multiwrite(logfiles,",")
			end
			iter = iter + 1
			multiwrite(logfiles,string.format("\n	[ %d, %.7f, %.7f, %.7f, %.7f ]", iter, train_error, train_kl, test_error, test_kl))
		end
		multiwrite(logfiles,"\n]\n")
		for i,lf in ipairs(logfiles) do lf:close() end
	end
end

no_globals(main)
