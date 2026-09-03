function cfg=setup_arhge_paths(data_root,result_root)
%SETUP_ARHGE_PATHS Configure package paths.
    release_root=fileparts(fileparts(mfilename('fullpath')));
    code_root=fullfile(release_root,'code');
    addpath(code_root);
    if nargin<1 || isempty(data_root), data_root=fullfile(release_root,'data'); end
    if nargin<2 || isempty(result_root)
        result_root=fullfile(release_root,'results');
    end
    cfg.release_root=release_root;
    cfg.code_root=code_root;
    cfg.data_root=data_root;
    cfg.result_root=result_root;
    if ~exist(cfg.result_root,'dir'), mkdir(cfg.result_root); end
end
