function [weights,diagnostics]=arhge_v8_geometry_weights(blocks,C,k)
%ARHGE_V8_GEOMETRY_WEIGHTS Label-free evidence weights for geometry blocks.
%
% Evidence combines agreement with the unweighted graph barycenter and
% the C-th normalized-affinity eigengap. The lower-bounded simplex prevents
% any geometry from disappearing. No class label is accepted by this API.

    M=numel(blocks);
    graphs=cell(M,1);
    gap=zeros(M,1);
    for m=1:M
        graphs{m}=normalized_knn_graph(blocks{m},k);
        gap(m)=spectral_gap(graphs{m},C);
    end
    barycenter=graphs{1}*0;
    for m=1:M, barycenter=barycenter+graphs{m}/M; end
    disagreement=zeros(M,1);
    for m=1:M
        disagreement(m)=norm(graphs{m}-barycenter,'fro')^2/ ...
            max(norm(graphs{m},'fro')^2,eps);
    end
    gap=max(gap,1e-6);
    evidence=gap./(disagreement+median(disagreement)+eps);
    weights=bounded_quality_weights(evidence,0.05);

    diagnostics.graphs=graphs;
    diagnostics.barycenter=barycenter;
    diagnostics.spectral_gap=gap;
    diagnostics.disagreement=disagreement;
    diagnostics.evidence_score=evidence;
    % Alias retained for compatibility with the output routine.
    diagnostics.reliability=evidence;
end

function W=normalized_knn_graph(Z,k)
    Z=double(Z);
    Z=bsxfun(@rdivide,Z,sqrt(sum(Z.^2,2))+eps);
    n=size(Z,1);
    k=min(k,n-1);
    similarity=Z*Z';
    similarity(1:n+1:end)=-inf;
    [value,index]=maxk(similarity,k,2);
    rows=repmat((1:n)',1,k);
    W=sparse(rows(:),index(:),max(value(:),0),n,n);
    W=max(W,W');
    degree=max(full(sum(W,2)),eps);
    D=spdiags(1./sqrt(degree),0,n,n);
    W=D*W*D;
    W=(W+W')/2;
end

function value=spectral_gap(W,C)
    n=size(W,1);
    % Deterministic block subspace iteration avoids the repeated-ARPACK heap
    % corruption observed in MATLAB R2019b and computes only the leading
    % eigenspace required for the C-th gap.
    width=min(C+6,n);
    index=(1:n)';
    Q=zeros(n,width);
    for j=1:width
        Q(:,j)=cos((index-0.5)*(j-1)*pi/n);
    end
    [Q,~]=qr(Q,0);
    shifted=0.5*(W+speye(n));
    for iter=1:35
        [Q,~]=qr(shifted*Q,0);
    end
    small=(Q'*W*Q);
    eigenvalues=sort(real(eig((small+small')/2)),'descend');
    if numel(eigenvalues)>C
        value=max(eigenvalues(C)-eigenvalues(C+1),0);
    else
        value=max(eigenvalues(end),0);
    end
end

function w=bounded_quality_weights(q,lower)
    q=max(q(:),eps);
    M=numel(q);
    lower=min(lower,(1-eps)/M);
    free=true(M,1);
    w=zeros(M,1);
    while true
        remaining=1-sum(~free)*lower;
        candidate=remaining*q(free)/sum(q(free));
        bad=candidate<lower;
        if ~any(bad)
            w(free)=candidate;
            w(~free)=lower;
            break;
        end
        index=find(free);
        free(index(bad))=false;
        if ~any(free)
            w=ones(M,1)/M;
            break;
        end
    end
end
