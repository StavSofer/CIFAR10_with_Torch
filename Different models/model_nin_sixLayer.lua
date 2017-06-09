-- Low level
model:add(cudnn.SpatialConvolution(3,128,3,3))
model:add(nn.SpatialBatchNormalization(128))
model:add(nn.ReLU(true))
-- 128*28*28
model:add(cudnn.SpatialConvolution(128,96,1,1))
model:add(nn.SpatialBatchNormalization(96))
model:add(nn.ReLU(true))
-- 96*28*28
model:add(cudnn.SpatialConvolution(96,64,1,1))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
-- 64*28*28
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 64*14*14

model:add(cudnn.SpatialConvolution(64,128,1,1))
model:add(nn.SpatialBatchNormalization(128))
model:add(nn.ReLU(true))
-- 128*14*14
model:add(cudnn.SpatialConvolution(128,128,1,1))
model:add(nn.SpatialBatchNormalization(128))
model:add(nn.ReLU(true))
-- 128*14*14

model:add(cudnn.SpatialConvolution(128,10,1,1))
model:add(nn.SpatialBatchNormalization(10))
model:add(nn.ReLU(true))
-- 10*14*14
model:add(cudnn.SpatialMaxPooling(4,4,2,2))
-- 10*6*6
model:add(cudnn.SpatialMaxPooling(5,5,2,2))
-- 10*1*1