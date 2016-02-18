% Use the original weights filter on the corresponding portion of the input
% Note: it is implemented for a specific case where the padding results in the
% 'same' size ouput as the input. Will not work for other paddings or
% strides. Also, unlike most of the other methods, obviously it cannot be
% adapted to a general (non-holed) convolution
%
% We have X, the input, Y, the convolution forward output. 
% Z is the output of the next layer with Y as input. Z is some general function over Y.
%
% For example say we have a 1x3 weights filter and X is 1x7 and we use a hole of 2 so the
% holed weights is 1x5. Assume we use padding such that Y is also 1x7. 
% We first convolve the corresponding local part of X, [X11,X13,X15,X17] with the original W to get Y1.
% We similarly perform the same for the second part of X, [X12,X14,X16] and W to get Y2.
% We repeat this process and fill in Y by these Yi's in the corresponding places.
% Each pixel of X only participates in one of these iterations and affect a single map Yi 
% (but can affect many pixels in this Yi). Therefore, the returned dZ/dX is the derivative for 
% this pixel and we just need to place it in its corresponding location.
%
% X11 contributes to both y11 (by w2, due to the padding) and to y13 (due to the hole, by w1). Therefore
% dZ/dX11 = dZ/dy11 * dy11/dX11 + dZ/dy13 * dy13/dX11
% In fact the actual equation is dZ/dX11 = dZ/dy11 * dy11/dX11 + dZ/dy12 * dy12/dX11 + dZ/dy13 * dy13/dX11 + ...
% however X11 doesn't contribute to any other element so the other y dervatives w.r.t X11 are zero.
% Similarly X13 contributes to y11 (by w3) and to y13 (by w2) and to y15 (by w3).
% Therefore dZ/dX13 = dZ/dy11 * dy11/dX13 + dZ/dy13 * dy13/dX13 + dZ/dy15 * dy15/dX13
% When calling vl_nnconv we supply the relevant dZ/dy = [dZ/dy11, dZ/dy13,
% dZ/dy15, dZ/dy17] as derOutput.
% Therefore the returned dZ/dX from the convolution of [X11,X13,X15,X17] with the original W are the 
% derivatives for these pixels and we just need to place them in their corresponding locations.
% For the derivative w.r.t the weights we see that
% dZ/dw11 = dZ/dy11 * dy11/dw11 + dZ/dy12 * dy12/dw11 + dZ/dy13 * dy13/dw11 + ...
% where dZ/dy11 * dy11/dw11 + dZ/dy13 * dy13/dw11 + dZ/dy15 * dy15/dw11 + dZ/dy17 * dy17/dw11 
% is returned from the first call, and the rest from the second call so we
% just need to sum them
function Y=fwd_holed_local(X,W,B,obj)
Y=zeros(size(X,1),size(X,2),size(W,4),size(X,4),'single');
if obj.isGPU
    Y=gpuArray(Y);
end
for yhole=1:obj.hole
    for xhole=1:obj.hole
        Y(yhole:obj.hole:end,xhole:obj.hole:end,:,:) = vl_nnconv(...
            X(yhole:obj.hole:end,xhole:obj.hole:end,:,:), W, B, ...
            'pad', ((size(W,1)-1)/2)*[1 1 1 1], ...
            'stride', 1);
    end
end