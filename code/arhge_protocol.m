function protocol=arhge_protocol()
%ARHGE_PROTOCOL Datasets, run identifiers and model parameters.
    protocol.datasets={'20Newsgroups','3Sources','BBC4view','BBCSport', ...
        'Handwritten2','Caltech101-7'};
    protocol.seeds=2036:2045;
    protocol.metrics={'ACC','NMI','Purity'};
    protocol.version='RHDG-DSNMF-1.0';
    protocol.parameters=arhge_parameters(protocol.datasets);
end
