% Split using both the output and rows
function Y=fwd_holed_batch_output_weights(X,W,B,obj)

% when true it is slightly faster than calling the batch weights function
INLINE_BATCH_WEIGHTS_CODE=true;

fh=obj.size(1);
fw=obj.size(2);
fd=obj.size(3);
ksz=obj.size(4);
new_szh=1+obj.hole*(fh-1);
new_szw=1+obj.hole*(fw-1);
KBATCHSZ=1024;
KBATCHSZ=min(KBATCHSZ,ksz);
if INLINE_BATCH_WEIGHTS_CODE
    holedW=zeros([1 new_szw fd KBATCHSZ],'single');
    
    if obj.isGPU
        holedW=gpuArray(holedW);
    end
end
kbatches=ceil(ksz/KBATCHSZ);
batchY=cell(kbatches,1);
if length(obj.pad)==1
    obj.pad=repmat(obj.pad,1,4);
end
pad_top=obj.pad(1);
pad_bottom=obj.pad(2);
M=size(X,1);
for k=1:kbatches
    rng=(k-1)*KBATCHSZ+1:min(k*KBATCHSZ,ksz);
    if INLINE_BATCH_WEIGHTS_CODE
        if length(rng)<KBATCHSZ
            holedW=zeros([1 new_szw fd length(rng)],'single');
        end
        if obj.isGPU
            holedW=gpuArray(holedW);
        end
        
        for rowi=1:size(W,1)
            row=(rowi-1)*obj.hole+1;
            holedW(1,1:obj.hole:new_szw,:,:)=W(rowi,:,:,rng);
            newbias=B(1,rng);
            
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
                curbias=newbias;
            end
            curY = vl_nnconv(...
                X(cur_start_row:cur_end_row,:,:,:), holedW, curbias, ...
                'pad', [cur_pad_top cur_pad_bottom obj.pad(3:4)], ...
                'stride', obj.stride, ...
                obj.opts{:}) ;
            if rowi==1
                totY=curY;
            else
                totY=totY+curY;
            end
        end
    else
        origsize=obj.size(4);
        obj.size(4)=length(rng);
        totY=fwd_holed_batch_weights(X,W(:,:,:,rng),B(rng),obj);
        obj.size(4)=origsize;
    end
    batchY{k} = totY;
    
end
Y =cat(3,batchY{:}); % seems faster than pre-allocating
