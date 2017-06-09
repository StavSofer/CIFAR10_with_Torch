require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

--[[
    Function that gets as input a test set and labels
    loads a trained network (model) and returns its error
]]

opt = lapp[[
   --model                     (default mymodel)          model-filename
   --aug                       (default false)            augmentation:true/false   
]]

-- Augmentation module
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
        if permutation[i] % 6 == 0 then
            -- hflip (always on)
            image.hflip(input[i]:float(), input[i]:float())
        end
        if permutation[i] % 6 == 1 then
            -- rotate
            if opt.aug then
                local deg = torch.uniform()*180
                image.rotate(input[i]:float(), input[i]:float(), (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
            end
        end
        if permutation[i] % 6 == 2 then
            -- vflip
            if opt.aug then
                image.vflip(input[i]:float(), input[i]:float())
            end
        end
      end
    end
    --self.output:set(input)
    self.output:set(input:cuda())
    return self.output
  end
end

function testError(data, labels)
    -- returns testLoss, testError, confusion

    --local vars
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0
    local batchSize = 128
    -- our criterion
    local criterion = nn.ClassNLLCriterion():cuda()

    -- forward inputs (with step=batchsize)
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        -- narrow data/labels
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        -- forward
        local y = model:forward(x)
        -- get error
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        -- update confusion matrix
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    -- compute average loss and error
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid

    return avgLoss, avgError, tostring(confusion)
end

local trainset = torch.load('../cifar.torch/cifar10-train.t7')
local testset = torch.load('../cifar.torch/cifar10-test.t7')

-- center the data set (normalize)

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


-- load model and invoke function
-- print error/loss
model = torch.load(opt.model)
-- create tensor to assign loss/err
loss = torch.Tensor(1)
err = torch.Tensor(1)

-- compute and print error using our function
loss, err, confusion = testError(testData,testLabels)

print('Test error: ' .. err, 'Test Loss: ' .. loss)