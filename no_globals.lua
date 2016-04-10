#!/usr/local/torch/install/bin/th

function no_globals(main)
	local before = { }
	for k, v in pairs(_G)
	do
		before[k] = true
	end
	main(arg)
	for k, v in pairs(_G)
	do
		if before[k] == nil
		then
			io.stderr:write(string.format("GLOBAL LEAK! WTF IS '%s'?! LITTER BUG!\n", k))
		end
	end
end
