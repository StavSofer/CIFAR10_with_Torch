require 'nn'

local model = nn.Sequential()


model:add(nn.SpatialConvolution(3,10,5,5))
model:add(nn.SpatialBatchNormalization(10))
model:add(nn.ReLU(true))
-- 10*28*28
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 10*14*14
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 10*7*7
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 10*3*3
model:add(nn.SpatialMaxPooling(2,2,2,2))

-- Back to size of 10
model:add(nn.View(10):setNumInputDims(3))


return model