function [F,diagnostics]=arhge_v6_embedding(model,X,C,seed)
%ARHGE_V6_EMBEDDING Deepest-semantic graph fusion for ARHGE.
%
% The factor block is selected by the same label-free geometry calibration
% as v5. Unlike v5's equal averaging of all hierarchy levels, v6 uses the
% deepest consensus graph, because it is coupled directly to the final
% shared representation H. No ground-truth label or extra hyperparameter is
% used.

    [~,Hcal,geometry]=arhge_v5_cluster(model.H,C,seed);
    Sdeep=model.Sc{end};
    U=spectral_representation(Sdeep,C);

    density=zeros(numel(X),1);
    for v=1:numel(X), density(v)=nnz(X{v})/numel(X{v}); end
    rho=mean(density)^2;

    A=Hcal';
    A=A*(sqrt(size(A,1))/max(norm(A,'fro'),eps));
    B=U';
    B=bsxfun(@rdivide,B,sqrt(sum(B.^2,2))+eps);
    F=[A sqrt(max(rho,0))*B]';

    diagnostics.geometry=geometry;
    diagnostics.view_density=density;
    diagnostics.graph_confidence=rho;
    diagnostics.output_graph='DeepestConsensusGraph';
    diagnostics.consensus_graph=Sdeep;
    diagnostics.spectral_embedding=U;
end

function U=spectral_representation(S,C)
    W=max(S,S');
    W=W-spdiags(diag(W),0,size(W,1),size(W,2));
    d=max(full(sum(W,2)),eps);
    A=spdiags(1./sqrt(d),0,numel(d),numel(d))*W* ...
        spdiags(1./sqrt(d),0,numel(d),numel(d));
    A=(A+A')/2;
    opts.tol=1e-6;
    opts.maxit=1000;
    opts.v0=ones(size(A,1),1)/sqrt(size(A,1));
    [V,~]=eigs(A,C,'la',opts);
    V=bsxfun(@rdivide,V,sqrt(sum(V.^2,2))+eps);
    U=V';
end
