function [labels,F,diagnostics]=arhge_v5_cluster(H,C,seed)
%ARHGE_V5_CLUSTER Label-free adaptive geometry calibration and clustering.
%
% Three deterministic geometries are evaluated:
%   Raw              - the nonnegative consensus representation;
%   SampleL2         - directional representation on the positive sphere;
%   CenteredSampleL2 - common-component removal followed by sphere mapping.
%
% Selection uses only mean silhouette and never accesses ground-truth labels.

    names={'Raw','SampleL2','CenteredSampleL2'};
    K=numel(names);
    candidates=cell(K,1);
    candidate_labels=cell(K,1);
    silhouette_score=-inf(K,1);
    balance_entropy=zeros(K,1);

    for k=1:K
        candidates{k}=calibrate(H,names{k});
        rng(seed,'twister');
        candidate_labels{k}=kmeans(candidates{k}',C,'Replicates',20, ...
            'MaxIter',1000,'EmptyAction','singleton');
        silhouette_score(k)=mean(silhouette(candidates{k}', ...
            candidate_labels{k},'sqeuclidean'));
        counts=accumarray(candidate_labels{k},1,[C 1]);
        prob=counts/max(sum(counts),1);
        balance_entropy(k)=-sum(prob(prob>0).*log(prob(prob>0)))/log(C);
    end

    % Silhouette is the sole selection criterion. Entropy is recorded as a
    % diagnostic so pathological one-cluster-dominant solutions are visible.
    [~,best]=max(silhouette_score);
    labels=candidate_labels{best};
    F=candidates{best};
    diagnostics.names=names;
    diagnostics.silhouette=silhouette_score;
    diagnostics.balance_entropy=balance_entropy;
    diagnostics.selected_index=best;
    diagnostics.selected_name=names{best};
end

function E=calibrate(H,name)
    H=double(H);
    switch name
        case 'Raw'
            E=H;
        case 'SampleL2'
            E=bsxfun(@rdivide,H,sqrt(sum(H.^2,1))+eps);
        case 'CenteredSampleL2'
            F=H';
            F=bsxfun(@minus,F,mean(F,1));
            F=bsxfun(@rdivide,F,sqrt(sum(F.^2,2))+eps);
            E=F';
        otherwise
            error('Unknown ARHGE-v5 geometry %s.',name);
    end
end
