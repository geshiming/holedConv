function newweights = get_holed_weights(weights,obj)
fh=size(weights,1);
fw=size(weights,2);
fd=size(weights,3);
k=size(weights,4);
new_szh=1+obj.hole*(fh-1);
new_szw=1+obj.hole*(fw-1);
newweights=zeros([new_szh new_szw fd k],'single');
if obj.isGPU
    newweights=gpuArray(newweights);
end
newweights(1:obj.hole:new_szh,1:obj.hole:new_szw,:,:)=weights;
end
