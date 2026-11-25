% ============================================================
% 文件名：hybrid_phase_diagram.m
% 探讨异质性玩家池中的最优 AI 投放策略 (固定 alpha_1=1, alpha_2=-1)
% 横轴：beta (Type 1 玩家占比)
% 纵轴：dc (社交容忍度阈值)
% ============================================================

clear; clc;

%% ===== 全局设置 & 参数 =====
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

% 固定参数
epsilon = 0.02;  % 两类玩家统一的自然回归概率
p0 = 0;
r = 0.3;         % AI 投放比例
nRuns = 3;       % Halton多起点寻优次数 (调大可防局部最优，但更慢)

% 异质性参数 (经典对抗)
alpha1 = 1;      % Type 1: 偏好势均力敌 (占比 beta)
alpha2 = -1;     % Type 2: 偏好越级挑战 (占比 1-beta)

% 遍历网格
betaVals = linspace(0, 1, 51);   
dcVals   = linspace(0, 5, 51);

%% ===== 初始化存储矩阵 =====
E_L_optimal  = nan(length(dcVals), length(betaVals));
AI_bestLevel = nan(length(dcVals), length(betaVals));

fprintf('=== 启动异质性 (Hybrid) 模型计算 ===\n');
tic;

%% ===== 主循环：遍历 beta 和 dc =====
for i = 1:length(dcVals)
    dc = dcVals(i);
    fprintf('进度: 第 %d/%d 行, dc = %.3f\n', i, length(dcVals), dc);
    
    for j = 1:length(betaVals)
        beta = betaVals(j);
        
        % 针对每一组 (beta, dc)，枚举 AI 等级 0~5
        best_fval_overall = inf;
        best_level_overall = NaN;
        
        for level = 0:5
            y_fixed = zeros(6,1);
            y_fixed(level+1) = 1;
            
            % 调用求解器
            [sol, fval] = solve_hybrid_core(alpha1, alpha2, beta, dc, epsilon, p0, r, y_fixed, nRuns);
            
            if ~isempty(sol) && fval < best_fval_overall
                best_fval_overall = fval;
                best_level_overall = level;
            end
        end
        
        % 记录最优结果
        E_L_optimal(i,j)  = -best_fval_overall;
        AI_bestLevel(i,j) = best_level_overall;
    end
end
toc;
fprintf('=== 计算完成，开始绘图... ===\n');

%% ===== 绘图：最优 AI 等级相图 =====
figure('Position', [100, 100, 700, 500]);
imagesc(betaVals, dcVals, AI_bestLevel);
set(gca,'YDir','normal');

% 自定义 0~5 级的颜色映射 (由浅入深的蓝色系)
cmap = [
    0.94 0.97 1.00;  % 0 - 极浅蓝 (接近白色，代表无AI干预或0级)
    0.75 0.87 0.98;  % 1 - 浅蓝
    0.51 0.75 0.93;  % 2 - 天蓝
    0.24 0.53 0.84;  % 3 - 中蓝
    0.11 0.35 0.68;  % 4 - 深蓝
    0.03 0.19 0.42   % 5 - 海军蓝/藏青 (极深蓝)
];
colormap(interp1(0:5, cmap, linspace(0,5,256)));

cb = colorbar;
cb.Ticks = 0:5; 
cb.TickLabels = {'Lv.0', 'Lv.1', 'Lv.2', 'Lv.3', 'Lv.4', 'Lv.5'};

xlabel('Proportion of Type 1 Players ($\beta$, $\alpha_1=1$)');
ylabel('Tolerance Threshold ($d_c$)');
title('Optimal AI Placement in Hybrid Population');


%% ============================================================
%% 本地函数区
%% ============================================================

function [bestSol, bestFval] = solve_hybrid_core(alpha1, alpha2, beta, dc, epsilon, p0, r, y_fixed, nRuns)
    % 生成多起点初值
    zCandidates = generate_halton_points(nRuns, beta);
    
    solverOpts = optimoptions('fmincon','Algorithm','sqp','Display','none',...
        'StepTolerance',1e-10,'OptimalityTolerance',1e-8,...
        'ConstraintTolerance',1e-8,'MaxIterations',2000);

    % 线性等式约束：Type 1 概率和为 beta，Type 2 概率和为 1-beta
    Aeq = zeros(2, 12);
    Aeq(1, 1:6) = 1;
    Aeq(2, 7:12) = 1;
    beq = [beta; 1 - beta];
    
    lb = zeros(12,1); 
    ub = ones(12,1);
    
    bestFval = inf;
    bestSol = [];
    
    for k = 1:size(zCandidates,1)
        z0 = zCandidates(k,:)';
        
        % 目标函数：最大化总体的 E[L]
        obj = @(x) -sum((0:5)' .* (x(1:6) + x(7:12))) + ...
            1000 * norm(nonlcon_check(x, alpha1, alpha2, dc, epsilon, p0, r, y_fixed))^2;
        
        [sol, fval, exitflag] = fmincon(obj, z0, [], [], Aeq, beq, lb, ub, [], solverOpts);
        
        if exitflag >= 0 && fval < bestFval
            bestFval = fval;
            bestSol = sol;
        end
    end
end

function ceq = nonlcon_check(x, alpha1, alpha2, dc, epsilon, p0, r, y_fixed)
    % 提取两种玩家的活跃部分
    x1_active = x(2:6);
    x2_active = x(8:12);
    y_active  = y_fixed(2:6);
    
    % 计算统一匹配池的实体总量
    active_eff = sum(x1_active) + sum(x2_active) + r * sum(y_active);
    
    % 计算每个等级在匹配池中的权重 wgt
    if active_eff < 1e-12
        wgt = ones(5,1)/5;
    else
        wgt = (x1_active + x2_active + r * y_active) / active_eff;
    end
    
    % 根据统一权重，分别生成 Type 1 和 Type 2 的转移矩阵
    Th1 = build_transition_matrix(alpha1, dc, epsilon, p0, wgt);
    Th2 = build_transition_matrix(alpha2, dc, epsilon, p0, wgt);
    
    % 稳态方程：左乘稳态分布应该等于自身
    ceq1 = (x(1:6)' * Th1 - x(1:6)')';
    ceq2 = (x(7:12)' * Th2 - x(7:12)')';
    
    % 拼接两个残差作为非线性约束的整体输出
    ceq = [ceq1; ceq2];
end

function Th = build_transition_matrix(alpha, dc, epsilon, p0, wgt)
    K = 5;
    Th = zeros(6);
    
    % 0级的复活概率
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
end

function seq = halton1d_local(n, base)
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

function z0_mat = generate_halton_points(nRuns, beta)
    % 为 12 维变量生成 Halton 序列初值并归一化
    basePrimes = [2,3,5,7,11,13, 17,19,23,29,31,37];
    X0 = zeros(nRuns, 12);
    for dim = 1:12
        X0(:,dim) = halton1d_local(nRuns, basePrimes(dim));
    end
    
    z0_mat = zeros(nRuns, 12);
    for k = 1:nRuns
        % 分别归一化到 beta 和 1-beta
        x1 = X0(k, 1:6)';
        x1 = (x1 / sum(x1)) * beta;
        
        x2 = X0(k, 7:12)';
        x2 = (x2 / sum(x2)) * (1 - beta);
        
        z0_mat(k, :) = [x1; x2]';
    end
end