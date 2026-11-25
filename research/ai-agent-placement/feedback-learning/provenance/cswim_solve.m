function [bestSol,bestFval,bestExitflag,bestOutput,bestLevel, ...
          EL_eachLevel, X_eachLevel] = solve(alpha, dc, epsilon, p0, nRuns, aiOpts, z0_hint)
% =============================================================
%  求解：AI 单等级 0~5 枚举 + 固定 r
%  完整版本，不删功能 + 新增保存每个等级的结果
% =============================================================

    if nargin < 5 || isempty(nRuns), nRuns = 1; end
    if nargin < 6 || isempty(aiOpts)
        aiOpts = struct('mode','fixed','r_value',0,'solver','local');
    end
    if nargin < 7, z0_hint = []; end

    bestFval = inf;
    bestSol = [];
    bestExitflag = [];
    bestOutput = [];
    bestLevel = NaN;

    % ====== 新增存储 ======
    EL_eachLevel = nan(6,1);   % E[L] for levels 0..5
    X_eachLevel  = nan(6,6);   % columns: 6 levels

    % ====== 初值（原逻辑）======
    zCandidates = [];
    basePrimes = [2,3,5,7,11,13];
    X0 = zeros(nRuns,6);
    for dim = 1:6
        X0(:,dim) = halton1d(nRuns, basePrimes(dim));
    end
    for k = 1:nRuns
        x0 = X0(k,:)'; x0 = x0/sum(x0);
        zCandidates = [zCandidates; x0'];
    end

    % ====== AI 等级枚举 0~5 ======
    for level = 0:5

        y_fixed = zeros(6,1);
        y_fixed(level+1) = 1;

        best_f_thisLevel = inf;
        best_x_thisLevel = [];

        for kk = 1:size(zCandidates,1)
            z0 = zCandidates(kk,:)';
            [sol,fval,exitflag,output] = solve_core(alpha,dc,epsilon,p0,z0,aiOpts,y_fixed);

            if isempty(sol), continue; end
            if fval < best_f_thisLevel
                best_f_thisLevel = fval;
                best_x_thisLevel = sol;
            end
        end

        % ====== 新增：保存每个 AI=level 的结果 ======
        if ~isempty(best_x_thisLevel)
            EL_eachLevel(level+1) = -best_f_thisLevel;
            X_eachLevel(:,level+1) = best_x_thisLevel(1:6)/sum(best_x_thisLevel(1:6));
        end

        % ====== 原逻辑：记录最终最优等级 ======
        if best_f_thisLevel < bestFval
            bestFval = best_f_thisLevel;
            bestSol = best_x_thisLevel;
            bestLevel = level;
        end
    end

    bestExitflag = 1;
    bestOutput = [];
end

%% ============================================================
%% 核心求解器 solve_core（原样完整）
%% ============================================================

function [sol,fval,exitflag,output] = solve_core(alpha, dc, epsilon, p0, z0, aiOpts, y_fixed)

    solverOpts = optimoptions('fmincon','Algorithm','sqp','Display','none',...
        'StepTolerance',1e-12,'OptimalityTolerance',1e-9,...
        'ConstraintTolerance',1e-10,'FunctionTolerance',1e-10,...
        'MaxIterations',3000,'MaxFunctionEvaluations',200000);

    A = []; b = [];
    Aeq = ones(1,6); beq = 1;
    lb = zeros(6,1); ub = ones(6,1);

    r = aiOpts.r_value;

    obj = @(x) -sum((0:5)'.*x) + ...
        1000 * constraint_penalty(x, alpha, dc, epsilon, p0, r, y_fixed);

    [sol,fval,exitflag,output] = fmincon(obj,z0,A,b,Aeq,beq,lb,ub,[],solverOpts);
end

%% ============================================================
%% penalty（原样完整）
%% ============================================================

function val = constraint_penalty(x, alpha, dc, epsilon, p0, r, y_fixed)
    [~, ceq] = nonlcon_check(x, alpha, dc, epsilon, p0, r, y_fixed);
    val = norm(ceq)^2;
end

%% ============================================================
%% 非线性稳态约束（原样完整）
%% ============================================================

function [c, ceq] = nonlcon_check(x, alpha, dc, epsilon, p0, r, y_fixed)

    K = 5;  % levels 1..5

    x_active = x(2:6);
    y_active = y_fixed(2:6);

    active_eff = sum(x_active) + r * sum(y_active);
    if active_eff < 1e-12
        wgt = ones(K,1)/K;
    else
        wgt = (x_active + r*y_active) / active_eff;
    end

    Th = zeros(6);

    % 0 → 1
    Th(1,1) = 1 - epsilon; 
    Th(1,2) = epsilon;

    for i = 1:K
        pbar = zeros(1,3);
        for j = 1:K
            d = abs(j-i);
            s = 1/(1+exp(-alpha*(d - dc)));
            pp = (1-p0)*(1-s);
            pm = (1-p0)*s;
            pz = p0;

            if i == K
                pp = 0;
                tot = pm + pz;
                if tot > 0
                    pm = pm/tot; pz = pz/tot;
                else
                    pz = 1; pm = 0;
                end
            end
            pbar = pbar + wgt(j)*[pm pz pp];
        end

        row = i+1;

        if i==1
            Th(row,1:3) = pbar;
        elseif i<K
            Th(row,row-1:row+1) = pbar;
        else
            Th(row,row-1:row) = pbar(1:2);
        end
    end

    ceq = (x' * Th - x')';
    c = [];
end

%% ============================================================
%% Halton 序列生成器（原样完整）
%% ============================================================

function seq = halton1d(n, base)
    seq = zeros(n,1);
    for i = 1:n
        f = 1; r = 0; j = i;
        while j > 0
            f = f / base;
            r = r + f * mod(j, base);
            j = floor(j / base);
        end
        seq(i) = r;
    end
end
