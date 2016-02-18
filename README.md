# HoledConv Layer in [MatConvNet] (http://www.vlfeat.org/matconvnet/)
This is a crude implementation in MatConvNet of the 'hole' algorithm described in [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs] (http://arxiv.org/abs/1412.7062).
The only implemented part is the 'hole' algorithm, without any CRF afterwards.
Note that the full (and more efficient) implementation by the original authors is available at the [DeepLab] (https://bitbucket.org/deeplab/deeplab-public/) site and this implementation is not related to them. 

Unlike the original implementation in Caffe (C++), this implementation is in MATLAB and based on the underlying MatConvNet convolutional code.
Therefore, while somewhat slower, this is a very simple and high-level code which can be easily ported to any other Deep Learning library such as Theano, Torch etc., using their respected convolutional code.

It could also be adapted for regular convolutions (without 'holes') that uses large filter, which cannot fit into the GPU memory at once.

On some small data-sets we have, this algorithm seems to provide better results than [Fully Convolutional Networks for Semantic Segmentation] (http://arxiv.org/abs/1411.4038), both the FCN-32s and FCN-8s.
However, when training and testing on the PASCAL VOC 2011 validation, using [MatConvNet-FCN] (https://github.com/vlfeat/matconvnet-fcn) we get slightly worse results than FCN-8s.

| Model                | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy |
|----------------------|-----------|---------|--------------------|----------------|
| FCN-32s              | RV-VOC11  | 60.61   | 89.65              | 75.14          |
| FCN-8s               | RV-VOC11  | 62.00   | 89.72              | 79.35          |
| hole (7x7)           | RV-VOC11  | 61.51   | 90.01              | 76.00          |
| hole (3x3, LargeFOV) | RV-VOC11  | running | running            | running        |

 
The results here are based on the MatConvNet-FCN code (not the original FCN/DeepLab models), using PASCAL VOC 2011 validation set with additional segmentation (setting `opts.vocAdditionalSegmentations` to true).
Two experiments were adopted from the paper. The first is DeepLab-7x7 and the second is DeepLab-LargeFOV. They simply differ in the HoledConv parameters and network definitions.
As stated in the paper, the LargeFOV is much faster due to two differences - a smaller filter size (3x3 instead of 7x7) and less filters in the fully connected layers (1024 instead of 4096). 
When combined with CRF the authors state it performs on par as the 7x7 model. Without CRF it performs worse (as in the paper).
As seen, these results are different from the results stated in the original paper. The paper uses the 2012 validation set. Possibly different learning parameters were used as well.
However, the hole algorithm seems to be usefull in producing finer resolution results than FCN (even without CRF), though the differences are subtle.

Hope you find it usefull.

# Technical notes:
There are several implementations of the hole convolution in HoledConv. All of them result in the same output (disregarding numerical differences), but they vary in speed. They can be selected by setting the fwd_fn and bwd_fn properties in HoledConv. By default I'm using the version I've found to be fastest (`fwd_holed_local.m`), but a different one might be faster in other cases. The rest of the .m files can be discarded if needed.
There is also a simple `unit_test` method.

As seen in `fcnTrain_holed.m`, replacing the existing convolutional layer with a holed convolution layer is simple:
```Matlab
idx=net.getLayerIndex('conv5_1');
layer=net.layers(idx);
convBlock = HoledConv('size', layer.block.size, 'hasBias', layer.block.hasBias,'opts',layer.block.opts,'pad',layer.block.pad,'stride',layer.block.stride,'net',net) ;
convBlock.pad = 2;
convBlock.hole = 2;
layer.block=convBlock;
net.layers(idx)=layer;
