-- Low level
model:add(cudnn.SpatialConvolution(3,32,5,5))
model:add(nn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
-- 32*28*28
model:add(cudnn.SpatialConvolution(32,32,1,1))
model:add(nn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
-- 32*28*28
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 32*14*14


-- Mid level
model:add(cudnn.SpatialConvolution(32,64,3,3))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialConvolution(64,64,1,1))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
-- 64*14*14
model:add(cudnn.SpatialConvolution(64,32,1,1))
model:add(nn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
-- 32*14*14
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 32*7*7



-- Back to low level
model:add(cudnn.SpatialConvolution(32,64,3,3))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
-- 64*5*5
model:add(cudnn.SpatialConvolution(64,64,1,1))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
-- 64*5*5
model:add(cudnn.SpatialAveragePooling(4,4,2,2))
-- 64*1*1
model:add(nn.Dropout())


model:add(cudnn.SpatialConvolution(64,10,1,1))
model:add(nn.SpatialBatchNormalization(10))
model:add(nn.ReLU(true))
-- 10*1*1

-- Back to size of 10
model:add(nn.View(10):setNumInputDims(3))