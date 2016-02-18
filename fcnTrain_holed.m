function fcnTrain_holed(varargin)
%FNCTRAIN Train FCN model using MatConvNet

matconvnet_root='~/code/3rd/matconvnet';
matconvnet_fcn_root='~/code/3rd/a/matconvnet-fcn-master';

run(fullfile(matconvnet_root,'matlab/vl_setupnn')) ;
addpath(fullfile(matconvnet_root,'examples')) ;
addpath(matconvnet_fcn_root);

dataDir=fullfile(matconvnet_fcn_root,'data');

opts.holedConv.largeFOV=false;
opts.holedConv.fewer_filters_in_fc=true;

% experiment and data paths
if opts.holedConv.largeFOV
    opts.expDir = fullfile(dataDir,'holed_largeFOV-voc11') ;
else
    opts.expDir = fullfile(dataDir,'holed-voc11') ;
end
opts.dataDir = fullfile(dataDir,'newvoc11') ;
opts.modelType = 'fcn32s' ;
opts.sourceModelPath = fullfile(dataDir,'models/imagenet-vgg-verydeep-16.mat') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = true ;

opts.numFetchThreads = 1 ; % not used yet

% training options (SGD)
opts.train = struct([]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

trainOpts.batchSize = 20 ;
trainOpts.numSubBatches = 10 ;
trainOpts.continue = true ;
trainOpts.gpus = [1] ;
trainOpts.prefetch = true ;
trainOpts.expDir = opts.expDir ;
trainOpts.learningRate = 0.0001 * ones(1,50) ;
trainOpts.numEpochs = numel(trainOpts.learningRate) ;
opts.train=trainOpts;
% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get PASCAL VOC 12 segmentation dataset plus Berkeley's additional
% segmentations
if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
else
    imdb = vocSetup('dataDir', opts.dataDir, ...
        'edition', opts.vocEdition, ...
        'includeTest', false, ...
        'includeSegmentation', true, ...
        'includeDetection', false) ;
    if opts.vocAdditionalSegmentations
        imdb = vocSetupAdditionalSegmentations(imdb, 'dataDir', opts.dataDir) ;
    end
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% Get training and test/validation subsets
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;

% Get dataset statistics
if exist(opts.imdbStatsPath)
    stats = load(opts.imdbStatsPath) ;
else
    stats = getDatasetStatistics(imdb) ;
    save(opts.imdbStatsPath, '-struct', 'stats') ;
end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

% Get initial model from VGG-VD-16
net = fcnInitializeModel('sourceModelPath', opts.sourceModelPath) ;
if any(strcmp(opts.modelType, {'fcn16s', 'fcn8s'}))
    % upgrade model to FCN16s
    net = fcnInitializeModel16s(net) ;
end
if strcmp(opts.modelType, 'fcn8s')
    % upgrade model fto FCN8s
    net = fcnInitializeModel8s(net) ;
end
net.meta.normalization.rgbMean = stats.rgbMean ;
net.meta.classes = imdb.classes.name ;

% -------------------------------------------------------------------------
% Modify the network with holed convolution
% -------------------------------------------------------------------------

if opts.holedConv.largeFOV
    % 3x3, LargeFOV (no CRF). Similar to DeepLab-LargeFOV model
    % Pooling padding is slightly different
    
    idx=net.getLayerIndex('pool1');
    layer=net.layers(idx);
    layer.block.stride=[2 2];
    layer.block.pad=[0 1 0 1];
    layer.block.poolSize=[3 3];
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('pool2');
    layer=net.layers(idx);
    layer.block.stride=[2 2];
    layer.block.pad=[0 1 0 1];
    layer.block.poolSize=[3 3];
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('pool3');
    layer=net.layers(idx);
    layer.block.stride=[2 2];
    layer.block.pad=[0 1 0 1];
    layer.block.poolSize=[3 3];
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('pool4');
    layer=net.layers(idx);
    layer.block.stride=[1 1];
    layer.block.pad=[0 2 0 2];
    layer.block.poolSize=[3 3];
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('conv5_1');
    layer=net.layers(idx);
    convBlock = HoledConv('size', layer.block.size, 'hasBias', layer.block.hasBias,'opts',layer.block.opts,'pad',layer.block.pad,'stride',layer.block.stride,'net',net) ;
    convBlock.pad = 2;
    convBlock.hole = 2;
    layer.block=convBlock;
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('conv5_2');
    layer=net.layers(idx);
    convBlock = HoledConv('size', layer.block.size, 'hasBias', layer.block.hasBias,'opts',layer.block.opts,'pad',layer.block.pad,'stride',layer.block.stride,'net',net) ;
    convBlock.pad = 2;
    convBlock.hole = 2;
    layer.block=convBlock;
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('conv5_3');
    layer=net.layers(idx);
    convBlock = HoledConv('size', layer.block.size, 'hasBias', layer.block.hasBias,'opts',layer.block.opts,'pad',layer.block.pad,'stride',layer.block.stride,'net',net) ;
    convBlock.pad = 2;
    convBlock.hole = 2;
    layer.block=convBlock;
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('pool5');
    layer=net.layers(idx);
    layer.block.stride=[1 1];
    layer.block.pad=[0 2 0 2];
    layer.block.poolSize=[3 3];
    net.layers(idx)=layer;
    
    net.addLayer('pool5a', dagnn.Pooling('method','avg'),'x31','x31a')
    idx=net.getLayerIndex('pool5a');
    layer=net.layers(idx);
    layer.block.stride=[1 1];
    layer.block.pad=[0 2 0 2];
    layer.block.poolSize=[3 3];
    net.layers(idx)=layer;
    
    
    % decimate the learned weights filter in fc6 from 7x7 to 3x3
    f = net.getParamIndex('fc6f') ;
    orig_filter=net.params(f).value;

    % optionally use 1024 filters instead of 4096 in the fully connected layers
    if opts.holedConv.fewer_filters_in_fc
        out_range=1:4:4096;
    else
        out_range=1:4096;
    end
    decimated_filter=orig_filter(1:3:7,1:3:7,:,out_range);
    net.params(f).value=decimated_filter;
    b = net.getParamIndex('fc6b') ;
    net.params(b).value=net.params(b).value(out_range);
    
    net.setLayerInputs('fc6',{'x31a'});
    idx=net.getLayerIndex('fc6');
    layer=net.layers(idx);
    convBlock = HoledConv('size', layer.block.size, 'hasBias', layer.block.hasBias,'opts',layer.block.opts,'pad',layer.block.pad,'stride',layer.block.stride,'net',net) ;
    convBlock.hole = 12;
    convBlock.size = size(decimated_filter);
    convBlock.pad = floor((1+convBlock.hole*(size(decimated_filter,1)-1))/2).*[1 1 1 1];
    layer.block=convBlock;
    net.layers(idx)=layer;
    
    if opts.holedConv.fewer_filters_in_fc
        idx=net.getLayerIndex('fc7');
        layer=net.layers(idx);
        layer.block.size=[1 1 length(out_range) length(out_range)];
        net.layers(idx)=layer;
        f = net.getParamIndex('fc7f') ;
        orig_filter=net.params(f).value;
        decimated_filter=orig_filter(:,:,out_range,out_range);
        net.params(f).value=decimated_filter;
        b = net.getParamIndex('fc7b') ;
        net.params(b).value=net.params(b).value(out_range);
        
        idx=net.getLayerIndex('fc8');
        layer=net.layers(idx);
        layer.block.size(3)=length(out_range);
        net.layers(idx)=layer;
        f = net.getParamIndex('fc8f') ;
        orig_filter=net.params(f).value;
        decimated_filter=orig_filter(:,:,out_range,:);
        net.params(f).value=decimated_filter;
    end
else
    % 7x7 (no CRF). Similar to DeepLab-7x7 model
    
    idx=net.getLayerIndex('pool4');
    layer=net.layers(idx);
    layer.block.stride=[1 1];
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('conv5_1');
    layer=net.layers(idx);
    convBlock = HoledConv('size', layer.block.size, 'hasBias', layer.block.hasBias,'opts',layer.block.opts,'pad',layer.block.pad,'stride',layer.block.stride,'net',net) ;
    convBlock.pad = 2;
    convBlock.hole = 2;
    layer.block=convBlock;
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('conv5_2');
    layer=net.layers(idx);
    convBlock = HoledConv('size', layer.block.size, 'hasBias', layer.block.hasBias,'opts',layer.block.opts,'pad',layer.block.pad,'stride',layer.block.stride,'net',net) ;
    convBlock.pad = 2;
    convBlock.hole = 2;
    layer.block=convBlock;
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('conv5_3');
    layer=net.layers(idx);
    convBlock = HoledConv('size', layer.block.size, 'hasBias', layer.block.hasBias,'opts',layer.block.opts,'pad',layer.block.pad,'stride',layer.block.stride,'net',net) ;
    convBlock.pad = 2;
    convBlock.hole = 2;
    layer.block=convBlock;
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('pool5');
    layer=net.layers(idx);
    layer.block.stride=[1 1];
    layer.block.pad=[0 2 0 2];
    layer.block.poolSize=[3 3];
    net.layers(idx)=layer;
    
    idx=net.getLayerIndex('fc6');
    layer=net.layers(idx);
    convBlock = HoledConv('size', layer.block.size, 'hasBias', layer.block.hasBias,'opts',layer.block.opts,'pad',layer.block.pad,'stride',layer.block.stride,'net',net) ;
    convBlock.pad = [12 12 12 12];
    convBlock.hole = 4;
    layer.block=convBlock;
    net.layers(idx)=layer;
end

% upsample only by 8
net.removeLayer('deconv32');
N = 21;
upsz=8;
crop=upsz/2*ones(1,4);
filters = single(bilinear_u(upsz*2, N, N)) ;
net.addLayer('deconv8', ...
    dagnn.ConvTranspose(...
    'size', size(filters), ...
    'upsample', upsz, ...
    'crop', crop, ...
    'numGroups', N, ...
    'hasBias', false), ...
    'x38', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,21,'single') ;
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = numel(trainOpts.gpus) > 0 ;

% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), ...
    trainOpts, ....
    'train', train, ...
    'val', val, ...
    opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,'prefetch',nargout==0) ;
