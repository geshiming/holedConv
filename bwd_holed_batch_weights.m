% See the forward function for description
function [derX,derW,derB]=bwd_holed_batch_weights(X,W,B,derOutput,obj)
fh=obj.size(1);
fw=obj.size(2);
fd=obj.size(3);
ksz=obj.size(4);
new_szh=1+obj.hole*(fh-1);
new_szw=1+obj.hole*(fw-1);
derX=zeros(size(X),'single');
holedW=zeros([1 new_szw fd ksz],'single');
if obj.isGPU
    derX=gpuArray(derX);
    holedW=gpuArray(holedW);
end

if length(obj.pad)==1
    obj.pad=repmat(obj.pad,1,4);
end
bias=B;
pad_top=obj.pad(1);
pad_bottom=obj.pad(2);
M=size(X,1);

rowsW=size(W,1);
cur_derW=cell(rowsW,1);
for rowi=1:rowsW
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
    [cur_derX,cur_derW{rowi},cur_derB] = vl_nnconv(...
        X(cur_start_row:cur_end_row,:,:,:), holedW, curbias, derOutput, ...
        'pad', [cur_pad_top cur_pad_bottom obj.pad(3:4)], ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    
    derX(cur_start_row:cur_start_row+size(cur_derX,1)-1,:,:,:)=derX(cur_start_row:cur_start_row+size(cur_derX,1)-1,:,:,:)+cur_derX;
    %                 derW(rowi,:,:,:)=cur_derW(1,1:obj.hole:new_szw,:,:);
    cur_derW{rowi}=cur_derW{rowi}(1,1:obj.hole:new_szw,:,:);
    if rowi==1
        derB=cur_derB;
    end
end
% concatanation seems faster than pre-allocation and updates
derW=cat(1,cur_derW{:});
