require 'nn'

-- Network-in-Network
-- achieves 92% with BN and 88% without

local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end

-- size floor((width  + 2*padW - kW) / dW + 1)

Block(3,64,5,5,1,1,2,2)
-- 64*32*32
Block(64,128,1,1)
-- 128*32*32
Block(128,16,1,1)
-- 16*32*32
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
-- 16*16*16
model:add(nn.Dropout())

Block(16,32,3,3,1,1,2,2)
-- 32*18*18
Block(32,64,1,1)
-- 64*18*18
Block(64,32,1,1)
-- 32*18*18
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout())

Block(32,64,3,3,1,1,1,1)
-- 64*9*9
Block(64,64,1,1)
-- 64*9*9
Block(64,10,1,1)
-- 10*9*9
model:add(nn.SpatialMaxPooling(9,9,1,1):ceil())

model:add(nn.View(10*1*1))

for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
  v.weight:normal(0,0.05)
  v.bias:zero()
end

--print(#model:cuda():forward(torch.CudaTensor(1,3,32,32)))

return model