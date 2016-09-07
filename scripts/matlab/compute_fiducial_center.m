p1 = [-38.5595397949219, 17.0527439117432, -4.08868408203125];
p2 = [-21.6157417297363, 23.7062530517578, 29.9089965820313];
p3 = [15.3001670837402, 38.8559799194336, 11.5132446289063];
p4 = [13.46018409729, 16.7821426391602, -32.5468139648438];
p5 = [-13.0079488754272, -20.8673534393311, -27.1215209960938];
p6 = [-6.69712162017822, -35.203182220459, 13.0753784179688];
p7 = [15.2748985290527, -30.5433025360107, 14.9324340820313];
p8 = [35.845100402832, -9.78328227996826, -5.67303466796875];

%p1 = [-38.5595, 17.0527, -4.08868];
%p2 = [-21.6157, 23.7063, 29.909];
%p3 = [15.3002, 38.856, 11.5132];
%p4 = [13.4602, 16.7821, -32.5468];
%p5 = [-13.0079, -20.8674, -27.1215];
%p6 = [-6.69712, -35.2032, 13.0754];
%p7 = [15.2749, -30.5433, 14.9324];
%p8 = [35.8451, -9.78328, -5.67303];

p = [p1;p2;p3;p4;p5;p6;p7;p8];

%compute pairwise distances between all members of f
p2p_delta = [];
for i = 1:8
    pi = p(i,:);
    d = [];
    for j = 1:8
        pj = p(j,:);
        if i == j
            dist = [0];
        else
            dist = norm(pi - pj);
        end
        d = [d,dist];
    end
    p2p_delta = [p2p_delta;d];
end
p2p_delta = p2p_delta


% compute the distances of all points to initial guess
x = [0,0,0];    % initial guess of center
c_delta = [];
for i = 1:8
    pi = p(i,:);
    c_delta = [c_delta, norm(pi - x)];
end
c_delta = c_delta

perturbations = 1e6;
radius = 40.0335;
for i = 1:perturbations
    eps = (rand(1,3) - 0.5) / 5e3;
    c = x + eps;
    r_new = [];
    for j = 1:8
        pj = p(j,:);
        r = norm(pj - c);
        r_new = [r_new, r];
    end
    c_diff = abs(c_delta - radius);
    r_diff = abs(r_new - radius);
    err_prev = sum( c_diff );
    err_new = sum( r_diff );

    fail = false;
    if( err_new > err_prev )
        for j = 1:8
          if( r_diff(j) > c_diff(j) )
              fail = true;
              break;
          end
        end
%        c_delta = r_new;
%        x = c;
    end
    
    if( ~fail )
        c_delta = r_new;
        x = c;
    end
    
%    for j = 1:8
%        if( err_new(j) > err_prev(j) )
%            break;
%        end
%        c_delta = r_new;
%        x = c;
%    end
end

c = c
c_delta = [];
for i = 1:8
    pi = p(i,:);
    c_delta = [c_delta, norm(pi - x)];
end
c_delta = c_delta

%compute center
%k = [1,2,3,8];

%method 1
%A = [p(k(1),:) - p(k(2),:);
%     p(k(2),:) - p(k(3),:);
%     p(k(3),:) - p(k(4),:)]

%b = [ (norm(p(k(1)))^2 - (norm(p(k(2)))^2) / 2);
%      (norm(p(k(2)))^2 - (norm(p(k(3)))^2) / 2);
%      (norm(p(k(3)))^2 - (norm(p(k(4)))^2) / 2) ]

%c = A\b
%c2 = inv(A) * b

%method 2 - Beyer 1987

