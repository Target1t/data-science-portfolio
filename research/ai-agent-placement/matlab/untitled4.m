%% 机制分析：灵活对比不同 AI 等级（手动标注模式）
clear; clc;

% 1. 加载数据
X_all = readmatrix('X_fixedR03_allLevels.xlsx');    
EL_all = readmatrix('EL_fixedR03_allLevels.xlsx');  
% AI_grid = readmatrix('AI_bestLevel_grid.xlsx'); % 手动标注模式下可不使用

% 参数网格定义
alphaVals = linspace(-1, 1, 41);
dcVals    = linspace(0, 5, 51);
levels    = (0:5)'; 

%% 2. 选取分析点
targetAlpha = 1.0;
targetDc    = 2.5;

% 获取参数索引
[~, aIdx] = min(abs(alphaVals - targetAlpha)); 
[~, dIdx] = min(abs(dcVals - targetDc));
currRow = (aIdx-1) * 51 + dIdx;

%% 3. 【灵活配置】：指定等级及手动标注内容
AI_opt = 4;                     % 你想分析的第一个 AI 等级
label_opt = ' (Optimal)';       % 这里输入你想在图例后面显示的标注

AI_sub = 3;                     % 你想对比的第二个 AI 等级
label_sub = ' (Sub-optimal)';   % 这里输入你想在图例后面显示的标注

fprintf('正在分析点: alpha=%.2f, dc=%.2f\n', targetAlpha, targetDc);
fprintf('对比等级: AI=%d%s vs AI=%d%s\n', AI_opt, label_opt, AI_sub, label_sub);

%% 4. 数据提取
% 提取分布 (连续存储逻辑)
cols_opt = (AI_opt * 6 + 1) : (AI_opt * 6 + 6);
cols_sub = (AI_sub * 6 + 1) : (AI_sub * 6 + 6);

x_opt = X_all(currRow, cols_opt)';
x_sub = X_all(currRow, cols_sub)';

% 归一化
x_opt = x_opt / sum(x_opt);
x_sub = x_sub / sum(x_sub);

% 提取 Excel 中原始记录的 EL 值
EL_opt_val = EL_all(currRow, AI_opt + 1);
EL_sub_val = EL_all(currRow, AI_sub + 1);

% 计算贡献度
cont_opt = levels .* x_opt;
cont_sub = levels .* x_sub;

%% 5. 绘图 (保持专业 LaTeX 格式)
figure('Color', 'w', 'Position', [100, 100, 1000, 450]);

% --- 左图：人类分布形状对比 ---
subplot(1, 2, 1);
b1 = bar(levels, [x_opt, x_sub]);
grid on;
set(b1(1), 'FaceColor', [0 0.447 0.741], 'DisplayName', sprintf('AI Level %d%s', AI_opt, label_opt));
set(b1(2), 'FaceColor', [0.85 0.325 0.098], 'DisplayName', sprintf('AI Level %d%s', AI_sub, label_sub));
xlabel('Human Level $i$', 'Interpreter', 'latex');
ylabel('Proportion $x_i$', 'Interpreter', 'latex');
title(sprintf('Human Distribution Shape ($\\alpha=%.1f, d_c=%.1f$)', targetAlpha, targetDc), 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'latex');

% --- 右图：人类贡献与记录的 EL 对比 ---
subplot(1, 2, 2);
b2 = bar(levels, [cont_opt, cont_sub]);
grid on;
set(b2(1), 'FaceColor', [0 0.447 0.741], 'DisplayName', sprintf('$EL_{%d} = %.3f$%s', AI_opt, EL_opt_val, label_opt));
set(b2(2), 'FaceColor', [0.85 0.325 0.098], 'DisplayName', sprintf('$EL_{%d} = %.3f$%s', AI_sub, EL_sub_val, label_sub));
xlabel('Human Level $i$', 'Interpreter', 'latex');
ylabel('Contribution ($i \cdot x_i$)', 'Interpreter', 'latex');
title(sprintf('Contribution to $E[L]$ ($\\alpha=%.1f, d_c=%.1f$)', targetAlpha, targetDc), 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'latex');

sgtitle(sprintf('Mechanism Analysis: Comparing AI Level %d and %d', AI_opt, AI_sub), 'Interpreter', 'latex');