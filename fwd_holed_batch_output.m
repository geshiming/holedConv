% Split the weights filter into batches over its fourth dimension (number of output neurons/channels)
% Note that this is a general implementation and can be easily adapted to a
% regular (non-holed) convolution, for example a memory-heavy convolution,
% by using the original W rather than the generated holedW
%
% We have X, the input, Y, the convolution forward output with k (ksz) channels, meaning its fourth dimension
% has a length of k. Z is the output of the next layer with Y as input. Z is some general function over Y.
%
% Here Y is a 'vector', combined of a concatanation of Yi's and it is not the sum of them.
% For simplicity, assume we have a batch size of 1 so each Yi is a map: the convolution result of a 
% single weight filter, Wi := W(:,:,:,i) with X.
% For a single pixel Xr we want to find out the partial derivative dZ/dXr. It is equal by definition to
% dZ/dXr = dZ/dY1 * dY1/dXr + dZ/dY2 * dY2/dXr + ... + dZ/dYk * dYk/dXr
% where dZ/dYi is also a map for each output element of Yi (each output pixel).
% Here, we first convolve X with W1 to get Y1 and so only the derivative of Z w.r.t Y1 affects the derivative
% of Z w.r.t X, e.g we get dZ/dY1 * dY1/dXr, where dZ/dY1 is passed as an argument and dY1/dXr is calculated 
% by the underlying convolution implementation (vl_nnconv) before returning the product.
% Again note that dZ/dY1 is a map (matrix) and so is dY1/dXr.
% We similarly perform the same for X and W2 to get Y2 and the corresponding dZ/dY2 * dY2/dXr. 
% We repeat and summarize for all k to get dZ/dXr. In practice we evaluate dZ/dX for the entire input X, 
% not for each pixel Xr seperately dZ/dXr.
function Y=fwd_holed_batch_output(X,W,B,obj)
fh=obj.size(1);
fw=obj.size(2);
fd=obj.size(3);
ksz=obj.size(4);
new_szh=1+obj.hole*(fh-1);
new_szw=1+obj.hole*(fw-1);
KBATCHSZ=64;
KBATCHSZ=min(KBATCHSZ,ksz);
holedW=zeros([new_szh new_szw fd KBATCHSZ],'single');
if obj.isGPU
    holedW=gpuArray(holedW);
end
kbatches=ceil(ksz/KBATCHSZ);
curY=cell(kbatches,1);
for k=1:kbatches
    rng=(k-1)*KBATCHSZ+1:min(k*KBATCHSZ,ksz);
    if length(rng)<KBATCHSZ
        holedW=zeros([new_szh new_szw fd length(rng)],'single');
        if obj.isGPU
            holedW=gpuArray(holedW);
        end
    end
    holedW(1:obj.hole:new_szh,1:obj.hole:new_szw,:,:)=W(:,:,:,rng);
    newbias=B(1,rng);
    
    curY{k} = vl_nnconv(...
        X, holedW, newbias, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    %                 if k==1
    %                                 Y=gpuArray(zeros([size(curY{1},1) size(curY{1},2) ksz size(curY{1},4)],'single'));
    %
    %                 end
    %                 Y(:,:,rng,:)=curY{k};
end
Y =cat(3,curY{:}); % seems faster than pre-allocating
