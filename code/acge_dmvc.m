function [model, infos] = acge_dmvc(XX, rank_layers, in_options)
% ACGE_DMVC Deep Semi-NMF initializer used by RHDG-DSNMF.
%
% RHDG-DSNMF calls this routine only for graph-free pretraining, with beta,
% tau, mu, and kappa set to zero and graph refresh disabled after
% initialization. The objective below describes the routine's optional
% standalone graph-enabled mode; it is not the RHDG-DSNMF refinement
% objective, which is implemented in arhge_v8_core.m.
%
% Optional standalone objective:
%   sum_v alpha(v)^p ||X^v-Z_1^v...Z_L^v H||_F^2
% + beta sum_v,l alpha(v)^p omega(l)^q c_vl tr(H L_l^v H')
% + eta/2 sum_v,l ||S_l^v||_F^2
% + tau/2 sum_v,l>1 ||S_l^v-S_{l-1}^v||_F^2
% + mu/2 sum_v,l ||S_l^v-S_l^c||_F^2
% + kappa/2 ||H H'-I||_F^2.
%
% All views share H. The intermediate nonnegative H_l^v supply hierarchical
% neighbourhood candidates. c_vl balances graph/reconstruction scale.
% Graph rows are optimized on a sparse adaptive-neighbour support and are
% not periodically rebuilt by constructW.

    if nargin < 3 || isempty(in_options), in_options = struct(); end
    defaults.max_iter = 100;
    defaults.init_iter = 30;
    defaults.graph_k = 10;
    defaults.graph_update_freq = 5;
    defaults.beta = 0.1;
    defaults.eta = 0.1;
    defaults.tau = 0.1;
    defaults.mu = 0.1;
    defaults.kappa = 1e-3;
    defaults.p = 2;
    defaults.q = 2;
    defaults.ridge = 1e-6;
    defaults.min_weight = 0.02;
    defaults.view_temperature = 0.5;
    defaults.layer_temperature = 0.5;
    defaults.tol = 1e-5;
    defaults.verbose = 1;
    options = merge_struct(defaults, in_options);

    V = numel(XX);
    L = numel(rank_layers);
    n = size(XX{1}, 2);
    if options.p <= 1 || options.q <= 1
        error('options.p and options.q must be greater than 1.');
    end
    for v = 1:V
        if size(XX{v},2) ~= n, error('All views must contain the same samples.'); end
        XX{v} = normalize_columns(double(XX{v}));
    end

    alpha = ones(V,1) / V;
    omega = ones(L,1) / L;
    Z = cell(V,L);
    Hview = cell(V,L);

    % Greedy Semi-NMF pretraining for view-specific intermediate layers.
    for v = 1:V
        Hprev = XX{v};
        for l = 1:L
            [Z{v,l}, Hview{v,l}] = semi_nmf_init(Hprev, rank_layers(l), ...
                options.init_iter, options.ridge);
            Hprev = Hview{v,l};
        end
    end

    % Use the proven DMVC-style shared initialization. Row normalization
    % removes arbitrary Semi-NMF scale before the view representations are
    % fused; this was substantially stronger than refactorizing Xcat.
    Hshared = zeros(size(Hview{1,L}));
    for v = 1:V
        Hv = Hview{v,L};
        Hv = bsxfun(@rdivide,Hv,sqrt(sum(Hv.^2,2))+eps);
        Hshared = Hshared + alpha(v)*Hv;
    end
    Hshared = max(Hshared,eps);
    for v = 1:V
        Hview{v,L} = Hshared;
        Hprev = get_previous(XX, Hview, v, L);
        Z{v,L} = ridge_basis(Hprev, Hshared, options.ridge);
    end

    % Initial learned graphs and layer-wise consensus graphs.
    S = cell(V,L);
    Sc = cell(1,L);
    for l = 1:L
        for v = 1:V
            S{v,l} = learn_adaptive_graph(Hview{v,l}, [], [], options, omega(l));
        end
        Sc{l} = graph_consensus(S(:,l), alpha, options.p);
    end

    % Balance reconstruction and graph terms at their initial scales. NMF
    % factor scaling otherwise makes a conventional beta almost inert.
    graph_scale_ref = zeros(V,L);
    for v = 1:V
        P0 = Z{v,1};
        for l = 2:L, P0=P0*Z{v,l}; end
        recon0 = norm(XX{v}-P0*Hshared,'fro')^2;
        for l = 1:L
            graph_scale_ref(v,l) = min(recon0 / ...
                max(L*graph_energy(Hshared,S{v,l}),eps),1e5);
        end
    end

    infos.cost = zeros(options.max_iter,1);
    infos.alpha = zeros(options.max_iter,V);
    infos.omega = zeros(options.max_iter,L);
    infos.rel_change = zeros(options.max_iter,1);
    start_time = tic;

    for iter = 1:options.max_iter
        % Global deep reconstruction update, X ~= Z1*...*ZL*H.
        for v = 1:V
            for l = 1:L
                right = Hshared;
                for k = L:-1:l+1, right = Z{v,k}*right; end
                if l == 1
                    Z{v,l} = XX{v} * regularized_pinv(right,options.ridge);
                else
                    left = Z{v,1};
                    for k = 2:l-1, left = left*Z{v,k}; end
                    Z{v,l} = regularized_pinv(left,options.ridge) * XX{v} * ...
                        regularized_pinv(right,options.ridge);
                end
            end
        end

        % Update view-specific intermediate nonnegative representations.
        for v = 1:V
            for l = 1:L-1
                Hprev = get_previous(XX, Hview, v, l);
                Hcur = Hview{v,l};
                A = Z{v,l}' * Hprev;
                B = Z{v,l}' * Z{v,l};
                [Ap,An] = split_sign(A);
                [Bp,Bn] = split_sign(B);
                numerator = Ap + Bn * Hcur;
                denominator = An + Bp * Hcur;

                % The next-layer reconstruction couples adjacent depths.
                C = Z{v,l+1} * Hview{v,l+1};
                [Cp,Cn] = split_sign(C);
                numerator = numerator + Cp;
                denominator = denominator + Hcur + Cn;

                [W,D] = graph_parts(S{v,l});
                graph_scale = options.beta * omega(l)^options.q;
                numerator = numerator + graph_scale * Hcur * W;
                denominator = denominator + graph_scale * Hcur * D;
                Hview{v,l} = max(Hcur .* sqrt(max(numerator,eps) ./ ...
                    max(denominator,eps)), eps);
            end
        end

        % Shared deepest representation update, aggregated over all views.
        numH = zeros(size(Hshared));
        denH = zeros(size(Hshared));
        for v = 1:V
            av = alpha(v)^options.p;
            P = Z{v,1};
            for l = 2:L, P = P*Z{v,l}; end
            A = P' * XX{v};
            B = P' * P;
            [Ap,An] = split_sign(A);
            [Bp,Bn] = split_sign(B);
            numv = Ap + Bn*Hshared;
            denv = An + Bp*Hshared;
            for l = 1:L
                [W,D] = graph_parts(S{v,l});
                graph_scale = options.beta * omega(l)^options.q * graph_scale_ref(v,l);
                numv = numv + graph_scale*Hshared*W;
                denv = denv + graph_scale*Hshared*D;
            end
            numH = numH + av*numv;
            denH = denH + av*denv;
        end
        if options.kappa > 0
            numH = numH + options.kappa * Hshared;
            denH = denH + options.kappa * (Hshared*Hshared')*Hshared;
        end
        Hshared = max(Hshared .* sqrt(max(numH,eps) ./ max(denH,eps)), eps);
        for v = 1:V, Hview{v,L} = Hshared; end

        % Joint adaptive graph update with cross-layer inheritance and
        % view-consensus guidance.
        if iter == 1 || mod(iter, options.graph_update_freq) == 0
            Sold = S;
            Scold = Sc;
            for l = 1:L
                for v = 1:V
                    if l > 1, layer_prior = Sold{v,l-1}; else, layer_prior = []; end
                    S{v,l} = learn_adaptive_graph(Hview{v,l}, layer_prior, ...
                        Scold{l}, options, omega(l));
                end
                Sc{l} = graph_consensus(S(:,l), alpha, options.p);
            end
        end

        % Adaptive layer reliability.
        layer_loss = zeros(L,1);
        for l = 1:L
            for v = 1:V
                layer_loss(l) = layer_loss(l) + alpha(v)^options.p * ...
                    graph_scale_ref(v,l)*graph_energy(Hshared, S{v,l});
            end
        end
        omega_raw = max(layer_loss,eps).^(-1/(options.q-1));
        omega = stabilized_simplex_weight(omega_raw, options.min_weight);

        % Adaptive view reliability from reconstruction and geometry.
        view_loss = zeros(V,1);
        for v = 1:V
            P = Z{v,1};
            for l = 2:L, P = P*Z{v,l}; end
            R = XX{v}-P*Hshared;
            view_loss(v) = sum(R(:).^2);
            for l = 1:L
                view_loss(v) = view_loss(v) + ...
                    options.beta * omega(l)^options.q * graph_scale_ref(v,l) * ...
                    graph_energy(Hshared, S{v,l});
            end
        end
        alpha_raw = max(view_loss,eps).^(-1/(options.p-1));
        alpha = stabilized_simplex_weight(alpha_raw, options.min_weight);

        infos.cost(iter) = full_objective(XX,Z,Hview,S,Sc,alpha,omega, ...
            graph_scale_ref,options);
        infos.alpha(iter,:) = alpha';
        infos.omega(iter,:) = omega';
        if iter > 1
            infos.rel_change(iter) = abs(infos.cost(iter)-infos.cost(iter-1)) / ...
                max(1,abs(infos.cost(iter-1)));
        else
            infos.rel_change(iter) = inf;
        end
        if options.verbose && (iter == 1 || mod(iter,10) == 0)
            fprintf('ACGE-DMVC iter %3d: obj=%.6e, rel=%.3e\n', ...
                iter, infos.cost(iter), infos.rel_change(iter));
        end
        if iter > 10 && infos.rel_change(iter) < options.tol, break; end
    end

    infos.cost = infos.cost(1:iter);
    infos.alpha = infos.alpha(1:iter,:);
    infos.omega = infos.omega(1:iter,:);
    infos.rel_change = infos.rel_change(1:iter);
    infos.time = toc(start_time);
    infos.iter = iter;
    model.Z = Z;
    model.Hview = Hview;
    model.H = Hshared;
    model.S = S;
    model.Sc = Sc;
    model.alpha = alpha;
    model.omega = omega;
    model.options = options;
    model.graph_scale = graph_scale_ref;
end

function out = merge_struct(defaults, supplied)
    out = defaults;
    names = fieldnames(supplied);
    for i = 1:numel(names), out.(names{i}) = supplied.(names{i}); end
end

function X = normalize_columns(X)
    X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1)) + eps);
end

function Hprev = get_previous(XX,Hview,v,l)
    if l == 1, Hprev = XX{v}; else, Hprev = Hview{v,l-1}; end
end

function Z = ridge_basis(V,H,ridge)
    Z = (V*H') / (H*H' + ridge*eye(size(H,1)));
end

function P = regularized_pinv(A,ridge)
    if size(A,1) >= size(A,2)
        P = (A'*A + ridge*eye(size(A,2))) \ A';
    else
        P = A' / (A*A' + ridge*eye(size(A,1)));
    end
end

function [Z,H] = semi_nmf_init(V,r,iters,ridge)
    n = size(V,2);
    H = max(rand(r,n),eps);
    for t = 1:iters
        Z = ridge_basis(V,H,ridge);
        A = Z'*V;
        B = Z'*Z;
        [Ap,An] = split_sign(A);
        [Bp,Bn] = split_sign(B);
        H = max(H .* sqrt(max(Ap+Bn*H,eps) ./ max(An+Bp*H,eps)),eps);
    end
    Z = ridge_basis(V,H,ridge);
end

function [P,N] = split_sign(M)
    P = (abs(M)+M)/2;
    N = (abs(M)-M)/2;
end

function S = learn_adaptive_graph(H,layer_prior,consensus,options,layer_weight)
    n = size(H,2);
    k = min(options.graph_k,n-1);
    norms = sum(H.^2,1);
    dist = max(0,bsxfun(@plus,norms',norms)-2*(H'*H));
    dist(1:n+1:end) = inf;
    sorted_dist = sort(dist,2,'ascend');
    distance_scale = median(reshape(sorted_dist(:,1:k),[],1));
    dist = dist / max(distance_scale,eps);
    rows = cell(n,1); vals = cell(n,1);
    for i = 1:n
        [~,ord] = sort(dist(i,:),'ascend');
        cand = ord(1:k);
        if ~isempty(layer_prior), cand = union(cand,find(layer_prior(i,:)>0)); end
        if ~isempty(consensus), cand = union(cand,find(consensus(i,:)>0)); end
        cand(cand==i) = [];
        if numel(cand) > 4*k
            [~,keep] = sort(dist(i,cand),'ascend');
            cand = cand(keep(1:4*k));
        end
        prior = zeros(1,numel(cand)); common = prior;
        if ~isempty(layer_prior), prior = full(layer_prior(i,cand)); end
        if ~isempty(consensus), common = full(consensus(i,cand)); end
        denom = options.eta + options.tau*(~isempty(layer_prior)) + ...
            options.mu*(~isempty(consensus));
        y = (options.tau*prior + options.mu*common - ...
            0.5*options.beta*layer_weight^options.q*dist(i,cand)) / max(denom,eps);
        s = project_simplex(y);
        nz = s > 1e-12;
        rows{i} = cand(nz);
        vals{i} = s(nz);
    end
    nnz_total = sum(cellfun(@numel,rows));
    ii = zeros(nnz_total,1); jj = ii; ss = ii; pos = 1;
    for i = 1:n
        c = numel(rows{i});
        if c > 0
            idx = pos:pos+c-1; ii(idx)=i; jj(idx)=rows{i}; ss(idx)=vals{i}; pos=pos+c;
        end
    end
    S = sparse(ii,jj,ss,n,n);
    S = (S+S')/2;
end

function x = project_simplex(y)
    y = y(:)';
    u = sort(y,'descend');
    sv = cumsum(u);
    rho = find(u > (sv-1)./(1:numel(u)),1,'last');
    if isempty(rho), x = ones(size(y))/numel(y); return; end
    theta = (sv(rho)-1)/rho;
    x = max(y-theta,0);
end

function Sc = graph_consensus(Scol,alpha,p)
    w = alpha.^p; w = w/sum(w);
    Sc = Scol{1}*0;
    for v = 1:numel(Scol), Sc = Sc + w(v)*Scol{v}; end
    Sc = (Sc+Sc')/2;
end

function [W,D] = graph_parts(S)
    W = max(S,S');
    D = spdiags(full(sum(W,2)),0,size(W,1),size(W,1));
end

function e = graph_energy(H,S)
    [W,D] = graph_parts(S);
    e = real(trace(H*(D-W)*H'));
    e = max(e,0);
end

function w = stabilized_simplex_weight(raw,min_weight)
    raw = raw(:)/max(sum(raw),eps);
    w = max(raw,min_weight);
    w = w/sum(w);
end

function value = full_objective(XX,Z,Hview,S,Sc,alpha,omega,graph_scale_ref,options)
    [V,L] = size(Z);
    value = 0;
    for v = 1:V
        P = Z{v,1};
        for l = 2:L, P=P*Z{v,l}; end
        R = XX{v}-P*Hview{v,L};
        subtotal = sum(R(:).^2);
        for l = 1:L
            subtotal = subtotal + options.beta*omega(l)^options.q*graph_scale_ref(v,l)* ...
                graph_energy(Hview{v,L},S{v,l});
            value = value + 0.5*options.eta*sum(nonzeros(S{v,l}).^2) + ...
                0.5*options.mu*norm(S{v,l}-Sc{l},'fro')^2;
            if l > 1
                value = value + 0.5*options.tau*norm(S{v,l}-S{v,l-1},'fro')^2;
            end
        end
        value = value + alpha(v)^options.p*subtotal;
    end
    H = Hview{1,L};
    value = value + 0.5*options.kappa*norm(H*H'-eye(size(H,1)),'fro')^2;
end
