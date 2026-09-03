function run_all_experiments(max_new_runs,data_root)
%RUN_ALL_EXPERIMENTS Run the six-dataset, ten-run experimental protocol.
    if nargin<1 || isempty(max_new_runs), max_new_runs=inf; end
    if nargin<2, data_root=[]; end
    cfg=setup_arhge_paths(data_root,[]);
    protocol=arhge_protocol();
    outdir=cfg.result_root;
    if ~exist(outdir,'dir'), mkdir(outdir); end
    rawfile=fullfile(outdir,'main_results.mat');
    D=numel(protocol.datasets); R=numel(protocol.seeds);
    scores=nan(D,R,3); seconds=nan(D,R); iterations=nan(D,R);
    increases=nan(D,R); errors=cell(D,R); new_runs=0;
    if exist(rawfile,'file')
        old=load(rawfile);
        if isfield(old,'version') && strcmp(old.version,protocol.version)
            if isfield(old,'scores'), scores=old.scores; end
            if isfield(old,'seconds'), seconds=old.seconds; end
            if isfield(old,'iterations'), iterations=old.iterations; end
            if isfield(old,'increases'), increases=old.increases; end
            if isfield(old,'errors'), errors=old.errors; end
        end
    end
    for d=1:D
        [X,y,meta]=load_mv_dataset(protocol.datasets{d},cfg.data_root);
        ranks=arhge_architecture(meta,2);
        options=struct('beta',protocol.parameters.beta(d), ...
            'mu',protocol.parameters.mu(d), ...
            'gamma',protocol.parameters.gamma(d),'verbose',0);
        raw=arhge_v7_raw_geometry(X,meta.classes,5);
        for r=1:R
            if isfinite(scores(d,r,1)), continue; end
            if strcmp(protocol.datasets{d},'Caltech101-7')
                % Use the dataset-specific Caltech seed schedule.
                factor_seed=70000+protocol.seeds(r);
            else
                factor_seed=10000*d+protocol.seeds(r);
            end
            cluster_seed=100000+factor_seed;
            rng(factor_seed,'twister');
            timer=tic;
            try
                [model,info]=arhge_v8_dmvc(X,ranks,meta.classes, ...
                    cluster_seed,options,raw);
                seconds(d,r)=toc(timer);
                score=score_clustering_labels(model.labels,y);
                scores(d,r,:)=[score.acc score.nmi score.purity];
                iterations(d,r)=info.iter;
                increases(d,r)=info.objective_increases;
                fprintf('%-15s run_id=%d %.6f %.6f %.6f %.2fs\n', ...
                    protocol.datasets{d},protocol.seeds(r), ...
                    score.acc,score.nmi,score.purity,seconds(d,r));
            catch ME
                seconds(d,r)=toc(timer);
                errors{d,r}=getReport(ME,'extended','hyperlinks','off');
                fprintf(2,'FAILED %s run_id=%d: %s\n', ...
                    protocol.datasets{d},protocol.seeds(r),ME.message);
            end
            checkpoint=struct('version',protocol.version, ...
                'datasets',{protocol.datasets},'seeds',protocol.seeds, ...
                'parameters',protocol.parameters,'scores',scores, ...
                'seconds',seconds,'iterations',iterations, ...
                'increases',increases,'errors',{errors});
            save(rawfile,'-struct','checkpoint','-v7.3');
            export_summary(outdir,protocol,scores,seconds,iterations,increases);
            new_runs=new_runs+1;
            if new_runs>=max_new_runs, return; end
        end
    end
end

function export_summary(outdir,protocol,scores,seconds,iterations,increases)
    rows={};
    for d=1:numel(protocol.datasets)
        for q=1:3
            x=squeeze(scores(d,:,q)); x=x(isfinite(x));
            if isempty(x), m=nan; s=nan; else, m=mean(x); s=std(x); end
            rows(end+1,:)={protocol.datasets{d},protocol.metrics{q}, ...
                m,s,numel(x)}; %#ok<AGROW>
        end
    end
    T=cell2table(rows,'VariableNames', ...
        {'Dataset','Metric','Mean','Std','CompletedRuns'});
    writetable(T,fullfile(outdir,'main_results.csv'));
    run_summary=table(sum(isfinite(scores(:,:,1)),'all'),sum(isfinite(seconds),'all'), ...
        sum(increases(:)>0,'omitnan'),max(iterations(:),[],'omitnan'), ...
        'VariableNames',{'CompletedRuns','TimedRuns', ...
        'RunsWithObjectiveIncrease','MaxIterations'});
    writetable(run_summary,fullfile(outdir,'run_summary.csv'));
end
