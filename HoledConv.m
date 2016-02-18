% HoledConv Layer in [MatConvNet] (http://www.vlfeat.org/matconvnet/)
% This is a crude implementation in MatConvNet of the 'hole' algorithm described in [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs] (http://arxiv.org/abs/1412.7062).
classdef HoledConv < dagnn.Conv
    properties
        hole = 1
        
        % for now we do not automatically detect if MatConvNet is running
        % using GPU. Adjust accordingly
        isGPU = true
        
        % print time, for deubg purposes
        time_execution = false
        
        % There are several implementations of the hole convolution. All of them result in the same output
        % (disregarding numerical differences), but they vary in speed. They can be selected by setting the
        % fwd_fn and bwd_fn properties. Replace fwd with bwd for the backward propagation function. Also see comments within each function
        % 1: fwd_holed_simple: simple (will most likely fail when the hole is large)
        % 2: fwd_holed_batch_output: split the filter into batches over its fourth dimension (number of output neurons/channels)
        % 3: fwd_holed_batch_weights: split the filter into rows over its first dimension
        % 4: fwd_holed_batch_output_weights: the above two approaches: batches output + rows (forward only)
        % 5: fwd_holed_local: use the original filter on the corresponding portion of the input
        fwd_fn = @fwd_holed_local;
        bwd_fn = @bwd_holed_local;
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            if ~obj.hasBias, params{2} = [] ; end
            
            if obj.time_execution
                tic;
            end
            
            outputs{1}=obj.fwd_fn(inputs{1},params{1},params{2},obj);
            
            if obj.time_execution
                if obj.isGPU
                    % without wait(gpuDevice) the reported times are incorrect!
                    wait(gpuDevice) ;
                end
                fprintf('%g fwd (%g %g %g %g)',toc,size(params{1},1),size(params{1},2),size(params{1},3),size(params{1},4));
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if ~obj.hasBias, params{2} = [] ; end
            
            if obj.time_execution
                tic;
            end
            
            [derInputs{1},derParams{1},derParams{2}] = obj.bwd_fn(inputs{1},params{1},params{2},derOutputs{1},obj);
            
            if obj.time_execution
                if obj.isGPU
                    % without wait(gpuDevice) the reported times are incorrect!
                    wait(gpuDevice) ;
                end
                fprintf('%g bwd (%g %g %g %g)',toc,size(params{1},1),size(params{1},2),size(params{1},3),size(params{1},4));
            end
        end
        
        function obj = HoledConv(varargin)
            obj.load(varargin) ;
        end
    end
    
    methods(Static)
        function unit_test(big_filter)
            if big_filter
                X=single(reshape(1:28^2,28,28));
                W=single(reshape(1:7^2,7,7));
                obj_info.hole=4;
            else
                X=single(reshape(1:7^2,7,7));
                W=single(reshape(1:3^2,3,3));
                obj_info.hole=2;
            end
            B=single(11);
            XM=(size(X,1));
            XN=(size(X,2));
            derOutput=single(reshape(1:XM*XN,XM,XN));
            obj_info.stride=1;
            holed_filter_height=(1+obj_info.hole*(size(W,1)-1));
            obj_info.pad=floor(holed_filter_height/2).*[1 1 1 1];
            obj_info.size=[size(W) 1 1];
            obj_info.isGPU=false;
            if obj_info.isGPU
                X=gpuArray(X);
                W=gpuArray(W);
                B=gpuArray(B);
                derOutput=gpuArray(derOutput);
            end
            obj_info.opts={};
            
            % forward tests
            fprintf('Running forward simple\n');
            Y_simple=fwd_holed_simple(X,W,B,obj_info);
            
            fprintf('Running forward batch output\n');
            Y_bo=fwd_holed_batch_output(X,W,B,obj_info);
            fprintf('Numerical errors: output %f\n',norm(Y_bo(:)-Y_simple(:)));
            
            fprintf('Running forward batch weights\n');
            Y_bw=fwd_holed_batch_weights(X,W,B,obj_info);
            fprintf('Numerical errors: output %f\n',norm(Y_bw(:)-Y_simple(:)));
            
            fprintf('Running forward batch output and weights\n');
            Y_bow=fwd_holed_batch_output_weights(X,W,B,obj_info);
            fprintf('Numerical errors: output %f\n',norm(Y_bow(:)-Y_simple(:)));
            
            fprintf('Running forward local\n');
            Y_l=fwd_holed_local(X,W,B,obj_info);
            fprintf('Numerical errors: output %f\n',norm(Y_l(:)-Y_simple(:)));
            
            % backward tests
            fprintf('Running backward simple\n');
            [derX_simple,derW_simple,derB_simple]=bwd_holed_simple(X,W,B,derOutput,obj_info);
            
            fprintf('Running backward batch output\n');
            [derX_bo,derW_bo,derB_bo]=bwd_holed_batch_output(X,W,B,derOutput,obj_info);
            fprintf('Numerical errors: input derivative %f, weights derivative %f, bias derivative %f\n',norm(derX_bo(:)-derX_simple(:)),norm(derW_bo(:)-derW_simple(:)),norm(derB_bo(:)-derB_simple(:)));
            
            fprintf('Running backward batch weights\n');
            [derX_bw,derW_bw,derB_bw]=bwd_holed_batch_weights(X,W,B,derOutput,obj_info);
            fprintf('Numerical errors: input derivative %f, weights derivative %f, bias derivative %f\n',norm(derX_bw(:)-derX_simple(:)),norm(derW_bw(:)-derW_simple(:)),norm(derB_bw(:)-derB_simple(:)));
            
            fprintf('Running backward local\n');
            [derX_l,derW_l,derB_l]=bwd_holed_local(X,W,B,derOutput,obj_info);
            fprintf('Numerical errors: input derivative %f, weights derivative %f, bias derivative %f\n',norm(derX_l(:)-derX_simple(:)),norm(derW_l(:)-derW_simple(:)),norm(derB_l(:)-derB_simple(:)));
        end
    end
    
end
