-- Cifar 10 task - HW2

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'gnuplot'

--  ****************************************************************
--  Helper functions, variables, classes
--  ****************************************************************


function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

function plotError(trainError, testError, title)
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError_hw2_77_another_try.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

local model = nn.Sequential()

-- data augmentation module
do
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end 

  function BatchFlip:updateOutput(input)
    if self.train then
      local permutation = torch.randperm(input:size(1))
      for i=1,input:size(1) do
        if permutation[i] % 3 == 0 then
            -- flip
            image.hflip(input[i]:float(), input[i]:float())
        end
        if permutation[i] % 3 == 1 then

        end
        if permutation[i] % 3 == 2 then

        end
      end
    end
    --self.output:set(input)
    self.output:set(input:cuda())
    return self.output
  end
end

--  ****************************************************************
--  Load data
--  ****************************************************************
local trainset = torch.load('../cifar.torch/cifar10-train.t7')
local testset = torch.load('../cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

print(trainData:size())
print(testData:size())

--  ****************************************************************
--  Normalize data
--  ****************************************************************

-- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
local redChannel = trainData[{ {}, {1}, {}, {}  }] 
print(#redChannel)
local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

--  ****************************************************************
--  Build model
--  ****************************************************************
--[[
owidth = floor((width +2*padW - kW) / dW + 1)
oheight = floor((height +2*padH - kH) / dH + 1)
--]]

model:add(nn.BatchFlip())

-- Low level
model:add(cudnn.SpatialConvolution(3,32,5,5)) -- 2400
model:add(nn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
-- 32*28*28
model:add(cudnn.SpatialConvolution(32,32,1,1)) -- 1024
model:add(nn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
-- 32*28*28
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 32*14*14


-- Mid level
model:add(cudnn.SpatialConvolution(32,64,3,3)) -- 18432
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialConvolution(64,64,1,1)) -- 4096
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
-- 64*14*14
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 64*7*7

model:add(cudnn.SpatialConvolution(64,64,1,1)) -- 4096
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
-- 64*7*7

model:add(nn.Dropout(0.3))

-- Back to low level
model:add(cudnn.SpatialConvolution(64,32,3,3)) -- 18432
model:add(nn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialConvolution(32,32,1,1)) -- 1024
model:add(nn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
-- 32*5*5
model:add(cudnn.SpatialAveragePooling(2,2,2,2))
-- 32*2*2
--model:add(nn.Dropout())

-- Back to size of 10
model:add(nn.View(32*2*2):setNumInputDims(3))
model:add(nn.Linear(32*2*2, #classes)) -- 1280
model:add(nn.LogSoftMax())

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()

w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

--  ****************************************************************
--  Train
--  ****************************************************************

local batchSize = 128
local optimState = {}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        --local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize):cuda()
        --local yt = labels:narrow(1, i, batchSize)
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.adam(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

---------------------------------------------------------------------

epochs = 300
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()

for e = 1, epochs do
	print('1')
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    print('2')
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    print('3')
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end

plotError(trainError, testError, 'Classification Error')