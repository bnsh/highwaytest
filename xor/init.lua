local torch = require 'torch'
require 'paths'

local xor = {}
local trainset = {
-- Make this three dimensional _just_ so that it looks like mnist format
	data=torch.ByteTensor({
		{{0,0}},
		{{0,1}},
		{{1,0}},
		{{1,1}}
	}),
	size=4,
	label=torch.ByteTensor({
		0,
		1,
		1,
		0
	})
}

function xor.traindataset()
	return trainset
end

function xor.testdataset()
	return trainset
end

return xor
