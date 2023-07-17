function [assignments,costs]= assign2DByGibbs(C,numIteration,k)
%%KBEST2DASSIGN Find the k lowest cost 2D assignments for the
%               two-dimensional assignment problem with a rectangular cost
%               matrix C.
%
%INPUTS: C A numRowXnumCol cost matrix. numRow = number of objects. numCol
%          = number of objects + number of measurements
%        numIteration: number of iterations used in Gibbs sampling
%        k The number >=1 of hypotheses to generate. If k is less than the
%          total number of unique hypotheses, then all possible hypotheses
%          will be returned.
%
%OUTPUTS: col4rowBest A numRowXk matrix where the entry in each element
%                     is an assignment of the element in that row to a
%                     column. 0 entries signify unassigned rows.
%            costs    A kX1 vector containing the sum of the values of the
%                     assigned elements in C for all of the hypotheses.

n = size(C,1);
m = size(C,2) - n;

assignments= zeros(n,numIteration);
costs= zeros(numIteration,1);

currsoln= m+1:m+n; %use all missed detections as initial solution
assignments(:,1)= currsoln;
costs(1)=sum(C(sub2ind(size(C),1:n,currsoln)));
for sol= 2:numIteration
    for var= 1:n
        tempsamp= exp(-C(var,:)); %grab row of costs for current association variable
        tempsamp(currsoln([1:var-1,var+1:end]))= 0; %lock out current and previous iteration step assignments except for the one in question
        idxold= find(tempsamp>0); tempsamp= tempsamp(idxold);
        [~,currsoln(var)]= histc(rand(1,1),[0;cumsum(tempsamp(:))/sum(tempsamp)]);
        currsoln(var)= idxold(currsoln(var));
    end
    assignments(:,sol)= currsoln;
    costs(sol)= sum(C(sub2ind(size(C),1:n,currsoln)));
end
[unique_assignments,I,~]= unique(assignments','rows');
assignments= unique_assignments';
costs= costs(I);

if length(costs) > k
    [costs, sorted_idx] = sort(costs);
    costs = costs(1:k);
    assignments = assignments(:,sorted_idx(1:k));
end

end