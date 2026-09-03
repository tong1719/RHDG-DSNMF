function [model,infos]=arhge_v8_dmvc(XX,rank_layers,C,cluster_seed, ...
        in_options,raw_geometry)
%ARHGE_V8_DMVC Train the objective-consistent core and apply its output head.
% Default options match the experimental implementation.
    if nargin<5 || isempty(in_options), in_options=struct(); end
    [model,infos]=arhge_v8_core(XX,rank_layers,in_options);
    if nargin<6, raw_geometry=[]; end
    model=arhge_v8_output_head(model,XX,C,cluster_seed,in_options,raw_geometry);
end
