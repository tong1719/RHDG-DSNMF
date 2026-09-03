function out=score_clustering_labels(labels,gnd)
%SCORE_CLUSTERING_LABELS ACC, NMI and purity for a partition.
    mapped=best_map(gnd,labels);
    out.acc=mean(mapped(:)==gnd(:));
    out.nmi=nmi_score(gnd,labels);
    out.purity=purity_score(gnd,labels);
    out.labels=labels;
end

function mapped=best_map(y,c)
    uy=unique(y);
    uc=unique(c);
    K=max(numel(uy),numel(uc));
    M=zeros(K);
    for i=1:numel(uy)
        for j=1:numel(uc), M(i,j)=sum(y==uy(i)&c==uc(j)); end
    end
    % MATCHPAIRS solves the exact global assignment in polynomial time.
    % This replaces factorial enumeration at K=10 while preserving ACC.
    pairs=matchpairs(M,-1,'max');
    perm=zeros(1,K);
    perm(pairs(:,1))=pairs(:,2);
    mapped=c;
    for i=1:numel(uy)
        j=perm(i);
        if j<=numel(uc), mapped(c==uc(j))=uy(i); end
    end
end

function z=nmi_score(a,b)
    [~,~,a]=unique(a);
    [~,~,b]=unique(b);
    n=numel(a);
    C=accumarray([a b],1);
    pa=sum(C,2)/n;
    pb=sum(C,1)/n;
    p=C/n;
    [I,J]=find(p>0);
    mi=0;
    for k=1:numel(I)
        mi=mi+p(I(k),J(k))*log(p(I(k),J(k))/(pa(I(k))*pb(J(k))));
    end
    ha=-sum(pa(pa>0).*log(pa(pa>0)));
    hb=-sum(pb(pb>0).*log(pb(pb>0)));
    z=mi/max(sqrt(ha*hb),eps);
end

function z=purity_score(gnd,labels)
%PURITY_SCORE Fraction assigned to the majority class in each cluster.
    [~,~,g]=unique(gnd(:));
    [~,~,c]=unique(labels(:));
    contingency=accumarray([c g],1);
    z=sum(max(contingency,[],2))/numel(g);
end
