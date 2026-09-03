function ranks=arhge_architecture(meta,depth)
%ARHGE_ARCHITECTURE Reproducible architecture rule for depth analysis.
% Depth 2 uses the architecture [min(50,d_min), C].
    if nargin<2, depth=2; end
    C=meta.classes;
    dmin=min(meta.dimensions);
    switch depth
        case 1
            candidates=C;
        case 2
            candidates=[min(50,dmin) C];
        case 3
            candidates=[min(100,dmin) min(50,dmin) C];
        otherwise
            error('Supported depths are 1, 2 and 3.');
    end
    ranks=[];
    for value=candidates
        if isempty(ranks) || value<ranks(end)
            ranks(end+1)=value; %#ok<AGROW>
        end
    end
    if ranks(end)~=C, ranks(end+1)=C; end
end
