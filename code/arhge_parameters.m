function parameters=arhge_parameters(datasets)
%ARHGE_PARAMETERS Label-free parameters used in the experiments.
    names={'20Newsgroups','3Sources','BBC4view','BBCSport', ...
        'Handwritten2','Caltech101-7'};
    beta =[1e-2 1e-1 1e-1 1e-1 1    1e-1]';
    mu   =[1e-3 1e-3 1e-2 1e-2 1e-3 1e-3]';
    gamma=[1e-1 1    1e-1 1e-1 1e-1 1e-1]';
    if nargin<1 || isempty(datasets), datasets=names; end
    parameters=struct('beta',nan(numel(datasets),1), ...
        'mu',nan(numel(datasets),1),'gamma',nan(numel(datasets),1));
    for d=1:numel(datasets)
        j=find(strcmp(names,datasets{d}),1);
        if isempty(j), error('No parameters are defined for %s.',datasets{d}); end
        parameters.beta(d)=beta(j);
        parameters.mu(d)=mu(j);
        parameters.gamma(d)=gamma(j);
    end
end
