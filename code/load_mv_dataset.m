function [XX,gnd,meta]=load_mv_dataset(name,data_root)
%LOAD_MV_DATASET Load one of the six multi-view benchmark datasets.
% DATA_ROOT may be the package data directory or its parent directory.
    if nargin<2 || isempty(data_root)
        release_root=fileparts(fileparts(mfilename('fullpath')));
        data_root=fullfile(release_root,'data');
    end
    if exist(fullfile(data_root,'data'),'dir')
        droot=fullfile(data_root,'data');
    else
        droot=data_root;
    end
    key=regexprep(lower(name),'[^a-z0-9]','');
    switch key
        case '20newsgroups', file='20newsgroups.mat';
        case '3sources', file='3sources.mat';
        case 'bbc4view', file='BBC4view_685.mat';
        case 'bbcsport', file='BBCSport2view_544.mat';
        case 'handwritten2', file='handwritten.mat';
        case 'caltech1017', file='Caltech101-7.mat';
        otherwise, error('Unknown benchmark: %s',name);
    end
    source_file=fullfile(droot,file);
    if ~exist(source_file,'file')
        error('Dataset file not found: %s',source_file);
    end
    S=load(source_file); [XX,gnd]=decode_dataset(S,key);
    gnd=double(gnd(:)); [~,~,gnd]=unique(gnd,'stable');
    for v=1:numel(XX)
        X=double(XX{v});
        if size(X,2)~=numel(gnd) && size(X,1)==numel(gnd), X=X'; end
        if size(X,2)~=numel(gnd), error('Sample count mismatch in view %d.',v); end
        X(~isfinite(X))=0; XX{v}=X;
    end
    meta.name=name; meta.n=numel(gnd); meta.views=numel(XX);
    meta.classes=numel(unique(gnd)); meta.dimensions=cellfun(@(x)size(x,1),XX);
    meta.source_file=source_file;
end
function [X,y]=decode_dataset(S,key)
    y=[]; X=[];
    for n={'gnd','Y','y','truth','label','labels','truelabel'}
        if isfield(S,n{1}), y=S.(n{1}); break; end
    end
    if iscell(y), y=y{1}; end
    for n={'X','XX','data','fea','views'}
        if isfield(S,n{1}) && iscell(S.(n{1})), X=S.(n{1}); break; end
    end
    if isempty(y)
        f=fieldnames(S);
        for i=1:numel(f)
            z=S.(f{i});
            if isnumeric(z)&&isvector(z)&&numel(z)>10, y=z; break; end
        end
    end
    if isempty(X)
        f=fieldnames(S); c={};
        for i=1:numel(f)
            z=S.(f{i}); if isnumeric(z)&&ismatrix(z)&&~isvector(z), c{end+1}=z; end %#ok<AGROW>
        end
        X=c;
    end
    if strcmp(key,'handwritten2') && numel(X)>2, X=X(1:2); end
    if isempty(X)||isempty(y), error('Could not decode dataset file.'); end
end
