function result=run_demo(dataset,seed,data_root)
%RUN_DEMO Run one complete RHDG-DSNMF experiment.
% Example:
%   addpath('code'); run_demo('BBCSport',2036)
    if nargin<1 || isempty(dataset), dataset='BBCSport'; end
    if nargin<2 || isempty(seed), seed=2036; end
    if nargin<3, data_root=[]; end
    cfg=setup_arhge_paths(data_root,[]);
    protocol=arhge_protocol();
    d=find(strcmp(protocol.datasets,dataset),1);
    if isempty(d), error('Unknown dataset: %s',dataset); end
    [X,y,meta]=load_mv_dataset(dataset,cfg.data_root);
    ranks=arhge_architecture(meta,2);
    p=protocol.parameters;
    options=struct('beta',p.beta(d),'mu',p.mu(d), ...
        'gamma',p.gamma(d),'verbose',1);
    if strcmp(dataset,'Caltech101-7')
        factor_seed=70000+seed;
    else
        factor_seed=10000*d+seed;
    end
    cluster_seed=100000+factor_seed;
    rng(factor_seed,'twister');
    timer=tic;
    [model,info]=arhge_v8_dmvc(X,ranks,meta.classes,cluster_seed,options);
    elapsed=toc(timer);
    score=score_clustering_labels(model.labels,y);
    result=struct('dataset',dataset,'seed',seed,'ranks',ranks, ...
        'ACC',score.acc,'NMI',score.nmi,'Purity',score.purity, ...
        'seconds',elapsed,'iterations',info.iter, ...
        'objective_increases',info.objective_increases, ...
        'model_version',model.model_version,'parameters',options, ...
        'fusion_weights',model.v8.fusion_weights, ...
        'gate_active',model.v8.gate_active);
    fprintf('\nRHDG-DSNMF demo result\n');
    fprintf('Dataset: %s | run ID: %d | ranks: %s\n', ...
        dataset,seed,mat2str(ranks));
    fprintf('ACC %.6f | NMI %.6f | Purity %.6f | %.2f s\n', ...
        result.ACC,result.NMI,result.Purity,elapsed);
    fprintf('Gate active: %d | fusion mass: [%.4f %.4f %.4f]\n', ...
        result.gate_active,result.fusion_weights);
    save(fullfile(cfg.result_root, ...
        sprintf('demo_%s_seed%d.mat',dataset,seed)),'result');
end
