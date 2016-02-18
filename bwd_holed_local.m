% See the forward function for description
function [derX,derW,derB]=bwd_holed_local(X,W,B,derOutput,obj)
derX=(zeros(size(X),'single'));
if obj.isGPU
    derX=gpuArray(derX);
end
for yhole=1:obj.hole
    for xhole=1:obj.hole
        [cur_derX,cur_derW,cur_derB] = vl_nnconv(...
            X(yhole:obj.hole:end,xhole:obj.hole:end,:,:), W, B, derOutput(yhole:obj.hole:end,xhole:obj.hole:end,:,:), ...
            'pad', ((size(W,1)-1)/2)*[1 1 1 1], ...
            'stride', 1);
        derX(yhole:obj.hole:end,xhole:obj.hole:end,:,:)=cur_derX;
        if yhole==1 && xhole==1
            derW=cur_derW;
            derB=cur_derB;
        else
            derW=derW+cur_derW;
            derB=derB+cur_derB;
        end
    end
end
