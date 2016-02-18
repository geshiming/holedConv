% Split the weights filter into rows. 
% More generally every convolution can be seperated into multipication and summaries operation.
% Note that this is a general implementation and can be easily adapted to a
% regular (non-holed) convolution, for example a memory-heavy convolution,
% by using the original W rather than the generated holedW
% TODO: also verify when stride does not equal 1
%
% We have X, the input, Y, the convolution forward output. 
% Z is the output of the next layer with Y as input. Z is some general function over Y.
%
% Here Y = Y1 + Y2 + ... + Yp where p is the number of rows in the (holed) W.
% Hence dY/dX = dY/dY1 * dY1/dX + dY/dY2 * dY2/dX + ... + dY/dYp * dYp/dX. Also since dY/dYi = 1 here we end
% up with dY/dX = dY1/dX + dY2/dX + ... + dYp/dX.
% Here we define Wi := W(i,:,:,:). We first convolve X (adjusted for padding) with W1 to get Y1.
% We pass dZ/dY as the output derivative to the convolution and the derivative w.r.t the input we'll 
% get is dZ/dY * dY1/dX where dY1/dX was evaluated by the underlying convolution implementation (vl_nnconv).
% We similarly perform the same for X and W2 to get Y2 and the corresponding dZ/dY * dY2/dX. 
% We repeat and summarize and get dZ/dY * dY1/dX + dZ/dY * dY2/dX + ... + dZ/dY * dYp/dX = 
% dZ/dY * (dY1/dX + dY2/dX + ... + dYp/dX) = dZ/dY * dY/dX as shown earlier.
function Y=fwd_holed_batch_weights(X,W,B,obj)
fh=obj.size(1);
fw=obj.size(2);
fd=obj.size(3);
ksz=obj.size(4);
new_szh=1+obj.hole*(fh-1);
new_szw=1+obj.hole*(fw-1);

holedW=zeros([1 new_szw fd ksz],'single');
if obj.isGPU
    holedW=gpuArray(holedW);
end

if length(obj.pad)==1
    obj.pad=repmat(obj.pad,1,4);
end
bias=B;
pad_top=obj.pad(1);
pad_bottom=obj.pad(2);
M=size(X,1);

for rowi=1:size(W,1)
    row=(rowi-1)*obj.hole+1;
    holedW(1,1:obj.hole:new_szw,:,:)=W(rowi,:,:,:);
    
    
    orig_bottom_conv_start_row=M+pad_bottom-new_szh+1;
    cur_end_row=orig_bottom_conv_start_row+(row-1);
    cur_start_row=row-pad_top;
    if cur_start_row>=1
        cur_pad_top=0;
    else
        cur_pad_top=-cur_start_row+1;
        cur_start_row=1;
    end
    
    if cur_end_row<=M
        cur_pad_bottom=0;
    else
        cur_pad_bottom=cur_end_row-M;
        cur_end_row=M;
    end
    if row>1
        curbias=[];
    else
        curbias=bias;
    end
    curY = vl_nnconv(...
        X(cur_start_row:cur_end_row,:,:,:), holedW, curbias, ...
        'pad', [cur_pad_top cur_pad_bottom obj.pad(3:4)], ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    if rowi==1
        Y=curY;
    else
        Y=Y+curY;
    end
end
