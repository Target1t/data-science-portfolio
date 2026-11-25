% ============================================================
% 文件名：phase_diagram_compare.m (最终三图三表 + 机制分析扩展存储)
% 功能：
%   保持原功能完全不变：
%       - 计算 E[L] @ r=0
%       - 计算 E[L] @ r=0.3 (AI 单等级枚举)
%       - 生成三张原图
%       - 生成三份 excel
%
%   新增：
%       - 保存 r=0.3 时 AI=0..5 每个等级的 E[L]
%       - 保存 r=0.3 时 AI=0..5 每个等级的最佳 X(6维向量)
%
% ============================================================

clear; clc;

%% ===== 全局 LaTeX 设置 =====
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultColorbarTickLabelInterpreter','latex');

%% ===== 固定参数 =====
epsilon = 0.02;
p0 = 0;
nRuns = 5;
alphaVals = linspace(-1, 1, 41);
dcVals    = linspace(0, 5, 51);

%% ===== 求解选项 =====
aiOpts_fixedR0  = struct('mode','fixed','r_value',0.0,'solver','deterministic');
aiOpts_fixedR03 = struct('mode','fixed','r_value',0.3,'solver','deterministic');

%% ===== 初始化 =====
E_fixedR0 = nan(length(dcVals), length(alphaVals));
E_fixedR03 = nan(length(dcVals), length(alphaVals));
AI_bestLevel = nan(length(dcVals), length(alphaVals));

X_fixedR0 = nan(length(dcVals), length(alphaVals), 6);
X_fixedR03 = nan(length(dcVals), length(alphaVals), 6);

ExitFlags_fixed = nan(length(dcVals), length(alphaVals));

% ========= 新增数据保存 (dc × alpha × 6) =========
EL_fixedR03_allLevels = nan(length(dcVals), length(alphaVals), 6);
X_fixedR03_allLevels  = nan(length(dcVals), length(alphaVals), 6, 6);

fprintf('=== 启动路径跟踪计算 (r=0 vs r=0.3) ===\n');

%% ===== 主循环 =====
for i = 1:length(dcVals)
    dc = dcVals(i);
    fprintf('计算第 %d/%d 行, dc = %.3f\n', i, length(dcVals), dc);

    prevSol_fixed = [];

    for j = 1:length(alphaVals)
        alpha = alphaVals(j);

        %% ======================
        %% ==== 情况 1：r = 0 ====
        %% ======================
        [bestSol1, fval1, exit1] = solve(alpha, dc, epsilon, p0, nRuns, aiOpts_fixedR0, prevSol_fixed);

        E_fixedR0(i,j) = -fval1;
        ExitFlags_fixed(i,j) = exit1;

        if ~isempty(bestSol1)
            X_fixedR0(i,j,:) = bestSol1(1:6) / sum(bestSol1(1:6));
            prevSol_fixed = bestSol1;
        else
            prevSol_fixed = [];
        end

        %% =============================================
        %% ==== 情况 2：r = 0.3（AI 单等级枚举版） ====
        %% =============================================
        [bestSol2, fval2, exit2, out2, bestLevel2, ...
            EL_levels, X_levels] = solve(alpha, dc, epsilon, p0, nRuns, aiOpts_fixedR03, []);

        E_fixedR03(i,j) = -fval2;
        AI_bestLevel(i,j) = bestLevel2;

        if ~isempty(bestSol2)
            X_fixedR03(i,j,:) = bestSol2(1:6)/sum(bestSol2(1:6));
        end

        % ========== 新增记录 ==========
        EL_fixedR03_allLevels(i,j,:)    = EL_levels;
        X_fixedR03_allLevels(i,j,:,:)   = X_levels;
    end
end

fprintf('=== 计算完成，开始绘图... ===\n');

%% ===== 绘图 1：E[L] @ r=0 =====
try
    E_fixedR0_smooth = imgaussfilt(E_fixedR0, 1.0);
catch
    E_fixedR0_smooth = E_fixedR0;
end

figure;
imagesc(alphaVals, dcVals, E_fixedR0_smooth);
set(gca,'YDir','normal');
colorbar;
xlabel('$\alpha$'); ylabel('$d_c$');
title('平均等级 E[L], 固定 r=0 (平滑)');

%% ===== 绘图 2：E[L] @ r=0.3 =====
figure;
imagesc(alphaVals, dcVals, E_fixedR03);
set(gca,'YDir','normal');
colorbar;
xlabel('$\alpha$'); ylabel('$d_c$');
title('平均等级 E[L], 固定 r=0.3');

%% ===== 绘图 3：AI 最优等级 =====
figure;
imagesc(alphaVals, dcVals, AI_bestLevel);
set(gca,'YDir','normal');
cmap = [
    1.0 1.0 1.0;
    0.8 0.9 1.0;
    0.6 0.8 1.0;
    0.4 0.6 0.9;
    0.2 0.4 0.8;
    0.0 0.2 0.6];
colormap(interp1(0:5, cmap, linspace(0,5,256)));

cb = colorbar;
cb.Ticks = 0:5; cb.TickLabels = {'0','1','2','3','4','5'};
xlabel('$\alpha$'); ylabel('$d_c$');
title('AI 最优聚集等级 (0~5)');

fprintf('=== 三张原图生成完毕 ===\n');

%% ===== 原 Excel 输出 1：X_fixedR0_grid_smooth.xlsx =====
cellData = cell(length(dcVals)+1, length(alphaVals)+1);
cellData{1,1} = 'dc \ alpha';
for j = 1:length(alphaVals)
    cellData{1, j+1} = sprintf('%.3f', alphaVals(j));
end
for i = 1:length(dcVals)
    cellData{i+1, 1} = sprintf('%.3f', dcVals(i));
end

for i = 1:length(dcVals)
    for j = 1:length(alphaVals)
        xvec = squeeze(X_fixedR0(i,j,:))';
        if any(isnan(xvec))
            cellData{i+1,j+1} = 'NaN';
        else
            cellData{i+1,j+1} = sprintf('[%.3f %.3f %.3f %.3f %.3f %.3f]', xvec);
        end
    end
end
writecell(cellData, 'X_fixedR0_grid_smooth.xlsx');

%% ===== 原 Excel 输出 2：X_fixedR03_grid.xlsx =====
cellData2 = cell(size(cellData));
cellData2(1,:) = cellData(1,:);
cellData2(:,1) = cellData(:,1);

for i = 1:length(dcVals)
    for j = 1:length(alphaVals)
        xvec2 = squeeze(X_fixedR03(i,j,:))';
        if any(isnan(xvec2))
            cellData2{i+1,j+1} = 'NaN';
        else
            cellData2{i+1,j+1} = sprintf('[%.3f %.3f %.3f %.3f %.3f %.3f]', xvec2);
        end
    end
end
writecell(cellData2, 'X_fixedR03_grid.xlsx');

%% ===== 原 Excel 输出 3：AI_bestLevel_grid.xlsx =====
cellLevel = cell(length(dcVals)+1,length(alphaVals)+1);
cellLevel{1,1} = 'dc \ alpha';
for j = 1:length(alphaVals)
    cellLevel{1,j+1} = sprintf('%.3f', alphaVals(j));
end
for i = 1:length(dcVals)
    cellLevel{i+1,1} = sprintf('%.3f', dcVals(i));
end
for i = 1:length(dcVals)
    for j = 1:length(alphaVals)
        val = AI_bestLevel(i,j);
        if isnan(val)
            cellLevel{i+1,j+1} = 'NaN';
        else
            cellLevel{i+1,j+1} = sprintf('%d', val);
        end
    end
end
writecell(cellLevel, 'AI_bestLevel_grid.xlsx');

%% ===== 新增 excel：EL_fixedR03_allLevels.xlsx =====
tmp = reshape(EL_fixedR03_allLevels, [], 6);
writematrix(tmp, 'EL_fixedR03_allLevels.xlsx');

%% ===== 新增 excel：X_fixedR03_allLevels.xlsx =====
tmp2 = reshape(X_fixedR03_allLevels, [], 36);
writematrix(tmp2, 'X_fixedR03_allLevels.xlsx');

fprintf('=== 全部流程完成 (原功能 + 新功能) ===\n');
