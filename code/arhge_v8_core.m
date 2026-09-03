function [model,infos] = arhge_v8_core(XX,rank_layers,in_options)
%ARHGE_V8_CORE Objective-consistent RHDG-DSNMF training.
%
% The model retains the Deep Semi-NMF decoder learned during graph-free
% pretraining and jointly refines the common nonnegative representation,
% hierarchical graphs, view weights, and layer weights. Its three searched
% parameters are:
%   beta  - hierarchical graph regularization;
%   mu    - cross-view graph consensus;
%   gamma - tied encoder/decoder consistency.
%
% Fixed mechanisms:
%   * l2,1 reconstruction is optimized by IRLS sample reliabilities;
%   * view/layer weights solve a lower-bounded simplex subproblem;
%   * every layer graph is regularized by the representation that builds it;
%   * graphs are refreshed on a support containing the previous graph,
%     current kNN, adjacent-layer graphs, and the consensus graph;
%   * a weak fixed proximal term protects the graph-free initialization.

    if nargin<3 || isempty(in_options), in_options=struct(); end
    d.pretrain_iter=100;
    d.init_iter=50;
    d.refine_iter=100;
    d.graph_k=5;
    d.graph_update_freq=5;
    d.beta=0.1;
    d.mu=0.1;
    d.gamma=0.1;
    d.eta=0.1;
    d.tau=0.1;
    d.kappa=1e-3;
    d.trust_ratio=0.1;
    d.p=2;
    % Option q is the layer-weight exponent nu in the manuscript.
    d.q=2;
    d.ridge=1e-5;
    d.irls_eps=1e-8;
    d.tol=1e-5;
    d.verbose=1;
    d.pretrained_base=[];
    d.disable_dynamic_graph=false;
    d.disable_view_adaptation=false;
    d.disable_layer_adaptation=false;
    d.disable_bidirectional=false;
    d.disable_robust=false;
    d.disable_graph=false;
    d.disable_safeguard=false;
    d.backbone_only=false;
    % Optional graph-history recording is disabled by default.
    d.record_graph_history=false;
    o=merge_struct(d,in_options);
    if o.p<=1 || o.q<=1, error('p and q must exceed one.'); end
    V=numel(XX);
    L=numel(rank_layers);
    n=size(XX{1},2);
    for v=1:V
        if size(XX{v},2)~=n, error('All views must share the sample axis.'); end
    end

    % Use the same graph-free Deep Semi-NMF pretraining for every ablation.
    if isempty(o.pretrained_base)
        bo=struct('max_iter',o.pretrain_iter,'init_iter',o.init_iter, ...
            'graph_k',o.graph_k,'graph_update_freq',o.pretrain_iter+1, ...
            'beta',0,'tau',0,'mu',0,'kappa',0,'ridge',o.ridge, ...
            'p',o.p,'q',o.q,'min_weight',0,'verbose',0);
        [base,binfo]=acge_dmvc(XX,rank_layers,bo);
    else
        base=o.pretrained_base;
        binfo=struct('cost',[],'time',0,'iter',0,'cached',true);
    end
    X=cell(size(XX));
    for v=1:V, X{v}=normalize_columns(double(XX{v})); end
    Z=base.Z;
    H0=max(base.H,eps);
    H=H0;

    if o.backbone_only
        model=base;
        model.H=H0;
        model.H0=H0;
        model.options=strip_cached_base(o);
        infos=binfo;
        infos.cost=[];
        infos.iter=0;
        return;
    end

    % Fixed deep decoders and tied-encoder targets.
    P=cell(V,1);
    T=cell(V,1);
    Tp=cell(V,1);
    Tn=cell(V,1);
    recon0=zeros(V,1);
    enc_scale=ones(V,1);
    graph_scale=ones(V,L);
    gram_level=0;
    for v=1:V
        P{v}=Z{v,1};
        for l=2:L, P{v}=P{v}*Z{v,l}; end
        recon0(v)=norm(X{v}-P{v}*H0,'fro')^2;
        Tv=P{v}'*X{v};
        for r=1:size(Tv,1)
            Tv(r,:)=Tv(r,:)*(norm(H0(r,:))/max(norm(Tv(r,:)),eps));
        end
        T{v}=Tv;
        [Tp{v},Tn{v}]=split_sign(Tv);
        enc0=norm(H0-Tv,'fro')^2;
        enc_scale(v)=min(recon0(v)/max(enc0,eps),1e4);
        gram_level=gram_level+mean(diag(P{v}'*P{v}))/V;
    end

    % Initial layer graphs use all pretrained representations. The deepest
    % graph will subsequently be rebuilt from the evolving shared H.
    S=cell(V,L);
    for v=1:V
        for l=1:L
            if l<L, R=base.Hview{v,l}; else, R=H; end
            S{v,l}=adaptive_affinity(R,o.graph_k);
        end
    end
    alpha=ones(V,1)/V;
    % Internal variable gate is the layer-weight matrix omega in the paper.
    gate=ones(V,L)/L;
    Sc=graph_consensus_layers(S,alpha,o.p);
    for v=1:V
        for l=1:L
            R0=layer_representation(base,H0,v,l,L);
            graph_scale(v,l)=min(recon0(v)/ ...
                max(L*graph_energy(R0,S{v,l}),eps),1e5);
        end
    end

    if o.disable_safeguard
        trust=0;
    else
        trust=o.trust_ratio*gram_level;
    end
    view_floor=0.05/V;
    layer_floor=0.05/L;
    sample_weight=cell(V,1);
    for v=1:V, sample_weight{v}=ones(1,n); end

    infos.cost=nan(o.refine_iter,1);
    infos.alpha=nan(o.refine_iter,V);
    infos.gate=nan(o.refine_iter,V,L);
    infos.rel_change=nan(o.refine_iter,1);
    infos.graph_change=nan(o.refine_iter,1);
    infos.sample_weight_range=nan(o.refine_iter,V,2);
    if o.record_graph_history
        infos.graph_history.iterations=0;
        infos.graph_history.consensus={Sc};
    end
    start=tic;

    for iter=1:o.refine_iter
        % IRLS weights are the exact majorization weights of the l2,1 loss.
        for v=1:V
            R=X{v}-P{v}*H;
            if o.disable_robust
                sample_weight{v}=ones(1,n);
            else
                sample_weight{v}=0.5./sqrt(sum(R.^2,1)+o.irls_eps);
            end
        end

        % Multiplicative Semi-NMF update of the shared representation.
        num=zeros(size(H));
        den=zeros(size(H));
        for v=1:V
            av=alpha(v)^o.p;
            qv=sample_weight{v};
            A=bsxfun(@times,P{v}'*X{v},qv);
            B=P{v}'*P{v};
            [Ap,An]=split_sign(A);
            [Bp,Bn]=split_sign(B);
            numv=Ap+bsxfun(@times,Bn*H,qv);
            denv=An+bsxfun(@times,Bp*H,qv);

            if ~o.disable_bidirectional && o.gamma>0
                c=o.gamma*enc_scale(v);
                numv=numv+c*Tp{v};
                denv=denv+c*(H+Tn{v});
            end
            if ~o.disable_graph && o.beta>0
                % Only the deepest graph is a function of the evolving
                % shared code H. Shallow graphs regularize their matching
                % pretrained layer representations in the stated objective.
                l=L;
                c=o.beta*gate(v,l)^o.q*graph_scale(v,l);
                [W,D]=graph_parts(S{v,l});
                numv=numv+c*H*W;
                denv=denv+c*H*D;
            end
            num=num+av*numv;
            den=den+av*denv;
        end
        if trust>0
            num=num+trust*H0;
            den=den+trust*H;
        end
        if o.kappa>0
            num=num+o.kappa*H;
            den=den+o.kappa*(H*H')*H;
        end
        H=max(H.*sqrt(max(num,eps)./max(den,eps)),eps);

        % Block-coordinate graph refresh. Previous edges remain feasible;
        % hence a graph update cannot be caused merely by changing support.
        graph_delta=0;
        graph_refreshed=false;
        if ~o.disable_graph && ~o.disable_dynamic_graph && ...
                (iter==1 || mod(iter,o.graph_update_freq)==0)
            Sold=S;
            Scold=Sc;
            for v=1:V
                for l=1:L
                    if l<L, R=base.Hview{v,l}; else, R=H; end
                    parent=[];
                    child=[];
                    if l>1, parent=S{v,l-1}; end
                    if l<L, child=Sold{v,l+1}; end
                    graph_coefficient=o.beta*gate(v,l)^o.q* ...
                        graph_scale(v,l);
                    S{v,l}=learn_hier_graph(R,Sold{v,l},parent,child, ...
                        Scold{l},o,graph_coefficient);
                    graph_delta=graph_delta+norm(S{v,l}-Sold{v,l},'fro')^2;
                end
            end
            Sc=graph_consensus_layers(S,alpha,o.p);
            graph_refreshed=true;
        end

        % Exact lower-bounded simplex updates prevent view/layer collapse.
        layer_loss=zeros(V,L);
        view_loss=zeros(V,1);
        for v=1:V
            for l=1:L
                Rg=layer_representation(base,H,v,l,L);
                layer_loss(v,l)=o.beta*graph_scale(v,l)* ...
                    graph_energy(Rg,S{v,l});
            end
            if ~o.disable_layer_adaptation
                gate(v,:)=bounded_inverse_weight(layer_loss(v,:), ...
                    o.q,layer_floor)';
            end
            R=X{v}-P{v}*H;
            if o.disable_robust
                view_loss(v)=norm(R,'fro')^2;
            else
                view_loss(v)=sum(sqrt(sum(R.^2,1)+o.irls_eps));
            end
            if ~o.disable_bidirectional
                view_loss(v)=view_loss(v)+o.gamma*enc_scale(v)* ...
                    norm(H-T{v},'fro')^2;
            end
            if ~o.disable_graph
                for l=1:L
                    view_loss(v)=view_loss(v)+gate(v,l)^o.q* ...
                        layer_loss(v,l)+0.5*o.eta*sum(nonzeros(S{v,l}).^2)+ ...
                        0.5*o.mu*norm(S{v,l}-Sc{l},'fro')^2;
                    if l>1
                        view_loss(v)=view_loss(v)+0.5*o.tau* ...
                            norm(S{v,l}-S{v,l-1},'fro')^2;
                    end
                end
            end
        end
        if ~o.disable_view_adaptation
            alpha=bounded_inverse_weight(view_loss,o.p,view_floor);
        end
        Sc=graph_consensus_layers(S,alpha,o.p);
        if o.record_graph_history && graph_refreshed
            infos.graph_history.iterations(end+1)=iter;
            infos.graph_history.consensus{end+1}=Sc;
        end

        infos.cost(iter)=objective_value(X,P,T,H,H0,base,S,Sc,alpha,gate, ...
            graph_scale,trust,o);
        infos.alpha(iter,:)=alpha';
        infos.gate(iter,:,:)=gate;
        infos.graph_change(iter)=sqrt(graph_delta);
        for v=1:V
            infos.sample_weight_range(iter,v,:)=[min(sample_weight{v}) ...
                max(sample_weight{v})];
        end
        if iter>1
            infos.rel_change(iter)=abs(infos.cost(iter)-infos.cost(iter-1))/ ...
                max(1,abs(infos.cost(iter-1)));
        else
            infos.rel_change(iter)=inf;
        end
        if o.verbose && (iter==1 || mod(iter,10)==0)
            fprintf('RHDG-DSNMF iter %3d obj %.6e rel %.3e graph %.3e\n', ...
                iter,infos.cost(iter),infos.rel_change(iter), ...
                infos.graph_change(iter));
        end
        if iter>=10 && infos.rel_change(iter)<o.tol && ...
                (o.disable_dynamic_graph || mod(iter,o.graph_update_freq)~=0)
            break;
        end
    end

    infos.cost=infos.cost(1:iter);
    infos.alpha=infos.alpha(1:iter,:);
    infos.gate=infos.gate(1:iter,:,:);
    infos.rel_change=infos.rel_change(1:iter);
    infos.graph_change=infos.graph_change(1:iter);
    infos.sample_weight_range=infos.sample_weight_range(1:iter,:,:);
    infos.time=toc(start)+binfo.time;
    infos.iter=iter;
    infos.pretrain=binfo;
    infos.objective_increases=sum(diff(infos.cost)> ...
        1e-8*max(1,abs(infos.cost(1:end-1))));

    model=base;
    model.H=H;
    model.H0=H0;
    model.Z=Z;
    model.P=P;
    model.encoder_target=T;
    model.S=S;
    model.Sc=Sc;
    model.alpha=alpha;
    model.gate=gate;
    model.graph_scale=graph_scale;
    model.sample_weight=sample_weight;
    model.trust=trust;
    model.options=strip_cached_base(o);
    model.model_version='ARHGE-v8.0-objective-consistent-core';
end

function out=merge_struct(d,s)
    out=d;
    names=fieldnames(s);
    for i=1:numel(names), out.(names{i})=s.(names{i}); end
end

function o=strip_cached_base(o)
    if isfield(o,'pretrained_base'), o.pretrained_base=[]; end
end

function X=normalize_columns(X)
    X=bsxfun(@rdivide,X,sqrt(sum(X.^2,1))+eps);
end

function [P,N]=split_sign(M)
    P=(abs(M)+M)/2;
    N=(abs(M)-M)/2;
end

function A=adaptive_affinity(H,k)
    Y=double(H');
    n=size(Y,1);
    k=min(k,n-1);
    d=max(bsxfun(@plus,sum(Y.^2,2),sum(Y.^2,2)')-2*(Y*Y'),0);
    A=sparse(n,n);
    for i=1:n
        [ds,idx]=sort(d(i,:),'ascend');
        idx=idx(2:k+2);
        ds=ds(2:k+2);
        denom=k*ds(end)-sum(ds(1:k));
        if denom<=eps
            w=ones(1,k)/k;
        else
            w=(ds(end)-ds(1:k))/denom;
        end
        A(i,idx(1:k))=w;
    end
end

function R=layer_representation(base,H,v,l,L)
    if l<L
        R=base.Hview{v,l};
    else
        R=H;
    end
end

function S=learn_hier_graph(H,Sold,parent,child,consensus,o,graph_coefficient)
    Y=double(H');
    n=size(Y,1);
    k=min(o.graph_k,n-1);
    norms=sum(Y.^2,2);
    dist=max(bsxfun(@plus,norms,norms')-2*(Y*Y'),0);
    dist(1:n+1:end)=inf;
    nearest=zeros(n,k);
    for i=1:n
        [~,ord]=sort(dist(i,:),'ascend');
        nearest(i,:)=ord(1:k);
    end
    rows=cell(n,1);
    vals=cell(n,1);
    for i=1:n
        cand=nearest(i,:);
        cand=union(cand,find(Sold(i,:)>0));
        if ~isempty(parent), cand=union(cand,find(parent(i,:)>0)); end
        if ~isempty(child), cand=union(cand,find(child(i,:)>0)); end
        if ~isempty(consensus), cand=union(cand,find(consensus(i,:)>0)); end
        cand(cand==i)=[];

        prior_sum=zeros(1,numel(cand));
        prior_count=0;
        if ~isempty(parent)
            prior_sum=prior_sum+full(parent(i,cand));
            prior_count=prior_count+1;
        end
        if ~isempty(child)
            prior_sum=prior_sum+full(child(i,cand));
            prior_count=prior_count+1;
        end
        common=zeros(1,numel(cand));
        if ~isempty(consensus), common=full(consensus(i,cand)); end
        denom=o.eta+o.tau*prior_count+o.mu*(~isempty(consensus));
        y=(o.tau*prior_sum+o.mu*common- ...
            0.5*graph_coefficient*dist(i,cand))/max(denom,eps);
        s=project_simplex(y);
        keep=s>1e-12;
        rows{i}=cand(keep);
        vals{i}=s(keep);
    end
    total=sum(cellfun(@numel,rows));
    ii=zeros(total,1);
    jj=zeros(total,1);
    ss=zeros(total,1);
    pos=1;
    for i=1:n
        c=numel(rows{i});
        if c>0
            ind=pos:pos+c-1;
            ii(ind)=i;
            jj(ind)=rows{i};
            ss(ind)=vals{i};
            pos=pos+c;
        end
    end
    S=sparse(ii,jj,ss,n,n);
end

function x=project_simplex(y)
    y=y(:)';
    u=sort(y,'descend');
    sv=cumsum(u);
    rho=find(u>(sv-1)./(1:numel(u)),1,'last');
    if isempty(rho)
        x=ones(size(y))/numel(y);
    else
        theta=(sv(rho)-1)/rho;
        x=max(y-theta,0);
    end
end

function w=bounded_inverse_weight(loss,p,lower)
    loss=max(loss(:),eps);
    m=numel(loss);
    lower=min(max(lower,0),(1-eps)/m);
    free=true(m,1);
    w=zeros(m,1);
    while true
        remaining=1-sum(~free)*lower;
        raw=loss(free).^(-1/(p-1));
        cand=remaining*raw/max(sum(raw),eps);
        bad=cand<lower;
        if ~any(bad)
            w(free)=cand;
            w(~free)=lower;
            break;
        end
        idx=find(free);
        free(idx(bad))=false;
        if ~any(free)
            w=ones(m,1)/m;
            break;
        end
    end
end

function Sc=graph_consensus_layers(S,alpha,p)
    [V,L]=size(S);
    Sc=cell(1,L);
    w=alpha(:).^p;
    w=w/max(sum(w),eps);
    for l=1:L
        Sc{l}=S{1,l}*0;
        for v=1:V, Sc{l}=Sc{l}+w(v)*S{v,l}; end
    end
end

function [W,D]=graph_parts(S)
    W=max((S+S')/2,0);
    D=spdiags(full(sum(W,2)),0,size(W,1),size(W,1));
end

function e=graph_energy(H,S)
    [W,D]=graph_parts(S);
    e=max(real(trace(H*(D-W)*H')),0);
end

function val=objective_value(X,P,T,H,H0,base,S,Sc,alpha,gate,scale,trust,o)
    V=numel(X);
    L=size(S,2);
    val=trust*norm(H-H0,'fro')^2+ ...
        0.5*o.kappa*norm(H*H'-eye(size(H,1)),'fro')^2;
    for v=1:V
        R=X{v}-P{v}*H;
        if o.disable_robust
            subtotal=norm(R,'fro')^2;
        else
            subtotal=sum(sqrt(sum(R.^2,1)+o.irls_eps));
        end
        if ~o.disable_bidirectional
            subtotal=subtotal+o.gamma*scale_encoder(X{v},P{v},H0,T{v})* ...
                norm(H-T{v},'fro')^2;
        end
        if ~o.disable_graph
            for l=1:L
                Rg=layer_representation(base,H,v,l,L);
                subtotal=subtotal+o.beta*gate(v,l)^o.q*scale(v,l)* ...
                    graph_energy(Rg,S{v,l})+ ...
                    0.5*o.eta*sum(nonzeros(S{v,l}).^2)+ ...
                    0.5*o.mu*norm(S{v,l}-Sc{l},'fro')^2;
                if l>1
                    subtotal=subtotal+0.5*o.tau* ...
                        norm(S{v,l}-S{v,l-1},'fro')^2;
                end
            end
        end
        val=val+alpha(v)^o.p*subtotal;
    end
end

function s=scale_encoder(X,P,H0,T)
    recon=norm(X-P*H0,'fro')^2;
    enc=norm(H0-T,'fro')^2;
    s=min(recon/max(enc,eps),1e4);
end
