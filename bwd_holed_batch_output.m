% See the forward function for description
function [derX,derW,derB]=bwd_holed_batch_output(X,W,B,derOutput,obj)
fh=obj.size(1);
fw=obj.size(2);
fd=obj.size(3);
ksz=obj.size(4);
new_szh=1+obj.hole*(fh-1);
new_szw=1+obj.hole*(fw-1);
derW=(zeros(size(W),'single'));
derB=(zeros(size(B),'single'));
derX=(zeros(size(X),'single'));
KBATCHSZ=448;
KBATCHSZ=min(KBATCHSZ,ksz);
holedW=zeros([new_szh new_szw fd KBATCHSZ],'single');
if obj.isGPU
    holedW=gpuArray(holedW);
    derW=gpuArray(derW);
    derB=gpuArray(derB);
    derX=gpuArray(derX);
end

kbatches=ceil(ksz/KBATCHSZ);

for k=1:kbatches
    rng=(k-1)*KBATCHSZ+1:min(k*KBATCHSZ,ksz);
    if length(rng)<KBATCHSZ
        holedW=gpuArray(zeros([new_szh new_szw fd length(rng)],'single'));
    end
    holedW(1:obj.hole:new_szh,1:obj.hole:new_szw,:,:)=W(:,:,:,rng);
    newbias=B(1,rng);
    
    [cur_derX, cur_derW, cur_derB] = vl_nnconv(...
        X, holedW, newbias, derOutput(:,:,rng,:), ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    
    derX=derX+cur_derX;
    derW(:,:,:,rng)=cur_derW(1:obj.hole:new_szh,1:obj.hole:new_szw,:,:);
    derB(:,rng)=cur_derB;
end
