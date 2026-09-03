function model=arhge_v8_output_head(model,XX,C,cluster_seed,options,raw_geometry)
%ARHGE_V8_OUTPUT_HEAD Label-free confidence-gated multi-geometry output.
% Default settings match the experimental protocol.
    if nargin<5 || isempty(options), options=struct(); end
    if nargin<6 || isempty(raw_geometry)
        raw_geometry=arhge_v7_raw_geometry(XX,C,model.options.graph_k);
    end
    prior=get_option(options,'fusion_prior',[1 4 1]);
    anchor_margin=get_option(options,'fusion_anchor_margin',1.2);
    requested_shrinkage=get_option(options,'fusion_shrinkage',0.75);
    density_threshold=get_option(options,'density_threshold',0.5);
    dense_multiplier=get_option(options,'dense_neighbor_multiplier',4);
    gate_rule=get_option(options,'fusion_gate_rule','confident_anchor');
    assert(numel(prior)==3 && all(prior>0),'fusion_prior must have 3 positive entries.');
    assert(anchor_margin>0,'fusion_anchor_margin must be positive.');
    assert(requested_shrinkage>=0 && requested_shrinkage<=1, ...
        'fusion_shrinkage must lie in [0,1].');

    [Fdeep,v6]=arhge_v6_embedding(model,XX,C,cluster_seed);
    blocks={scale_block(Fdeep'),raw_geometry.P,raw_geometry.G};
    [geometry_weight,weight_info]=arhge_v8_geometry_weights( ...
        blocks,C,model.options.graph_k);
    learned=sum(prior)*geometry_weight';
    switch gate_rule
        case 'confident_anchor'
            activate=learned(2)>=anchor_margin*max(learned([1 3]));
        case 'dominant_anchor'
            activate=learned(2)>=max(learned)-1e-12;
        case 'exceeds_prior'
            activate=learned(2)>prior(2);
        case 'always'
            activate=true;
        case 'never'
            activate=false;
        otherwise
            error('Unknown fusion_gate_rule: %s',gate_rule);
    end
    if activate
        shrinkage=requested_shrinkage;
    else
        shrinkage=0;
    end
    fusion_weight=(1-shrinkage)*prior+shrinkage*learned;
    Z=[sqrt(fusion_weight(1))*blocks{1} ...
       sqrt(fusion_weight(2))*blocks{2} ...
       sqrt(fusion_weight(3))*blocks{3}];

    rng(cluster_seed,'twister');
    if raw_geometry.mean_density>density_threshold
        neighbors=min(dense_multiplier*model.options.graph_k,size(Z,1)-1);
        labels=spectralcluster(Z,C,'NumNeighbors',neighbors);
        clusterer=sprintf('DenseSpectral%dK',dense_multiplier);
    else
        labels=kmeans(Z,C,'Replicates',20,'MaxIter',1000, ...
            'EmptyAction','singleton','Distance','correlation');
        neighbors=nan;
        clusterer='SparseCorrelationKMeans';
    end

    model.H_nmf=model.H;
    model.F_deep=Fdeep;
    model.F=Z';
    model.labels=labels;
    model.v6=v6;
    model.v8.raw_geometry=raw_geometry;
    model.v8.geometry_weight_info=weight_info;
    model.v8.geometry_weight=geometry_weight;
    model.v8.reliability_weight=geometry_weight;
    model.v8.fusion_weights=fusion_weight;
    model.v8.fusion_prior=prior;
    model.v8.shrinkage=shrinkage;
    model.v8.requested_shrinkage=requested_shrinkage;
    model.v8.gate_active=logical(activate);
    model.v8.gate_rule=gate_rule;
    model.v8.anchor_margin=anchor_margin;
    model.v8.density_threshold=density_threshold;
    model.v8.clusterer=clusterer;
    model.v8.output_neighbors=neighbors;
    model.model_version='RHDG-DSNMF-1.0';
end

function value=get_option(options,name,default)
    if isfield(options,name), value=options.(name); else, value=default; end
end

function B=scale_block(B)
    B=double(B);
    B=B*(sqrt(size(B,1))/max(norm(B,'fro'),eps));
end
