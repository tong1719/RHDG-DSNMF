function geometry=arhge_v7_raw_geometry(X,C,graph_k)
%ARHGE_V7_RAW_GEOMETRY Parameter-free raw-view residual geometry.
%
% Each normalized view contributes a rank-3C linear residual block and a
% graph_k cosine-neighbor graph. All blocks are created without labels.

    V=numel(X);
    n=size(X{1},2);
    pca_blocks=cell(V,1);
    W=sparse(n,n);
    density=zeros(V,1);
    for v=1:V
        density(v)=nnz(X{v})/numel(X{v});
        Y=double(X{v}');
        Y=bsxfun(@rdivide,Y,sqrt(sum(Y.^2,2))+eps);
        r=min([3*C,size(Y,1)-1,size(Y,2)]);
        [U,S,~]=svds(Y,r);
        pca_blocks{v}=scale_block(U*S);
        W=W+cosine_knn_graph(Y,graph_k)/V;
    end
    geometry.P=scale_block(horzcat(pca_blocks{:}));
    geometry.G=scale_block(spectral_embedding(W,C));
    geometry.raw_graph=W;
    geometry.view_density=density;
    geometry.mean_density=mean(density);
    geometry.pca_rank_multiplier=3;
    geometry.graph_k=graph_k;
end

function B=scale_block(B)
    B=double(B);
    B=B*(sqrt(size(B,1))/max(norm(B,'fro'),eps));
end

function W=cosine_knn_graph(Y,k)
    n=size(Y,1);
    S=Y*Y';
    S(1:n+1:end)=-inf;
    [values,index]=maxk(S,min(k,n-1),2);
    rows=repmat((1:n)',1,size(index,2));
    W=sparse(rows(:),index(:),max(values(:),0),n,n);
    W=max(W,W');
    W=W-spdiags(diag(W),0,n,n);
end

function U=spectral_embedding(W,C)
    W=max(W,W');
    W=W-spdiags(diag(W),0,size(W,1),size(W,2));
    degree=max(full(sum(W,2)),eps);
    A=spdiags(1./sqrt(degree),0,numel(degree),numel(degree))*W* ...
        spdiags(1./sqrt(degree),0,numel(degree),numel(degree));
    A=(A+A')/2;
    opts.tol=1e-6;
    opts.maxit=1000;
    opts.v0=ones(size(A,1),1)/sqrt(size(A,1));
    [U,~]=eigs(A,C,'la',opts);
    U=bsxfun(@rdivide,U,sqrt(sum(U.^2,2))+eps);
end
