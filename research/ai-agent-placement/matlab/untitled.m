%% 机制分析：基于最优结果 Excel 的直接出图
clear; clc;

% 1. 读取你准备好的两个核心结果文件
% AI_bestLevel_grid: 用于获取最优 AI 等级
AI_grid = readmatrix('AI_bestLevel_grid.xlsx'); 
% X_fixedR03_grid: 用于获取在该 AI 等级下，人类的最优分布
% 注意：这个文件里的单元格格式可能是字符串 "[0.123 0.456 ...]"
X_grid  = readtable('X_fixedR03_grid.xlsx', 'ReadVariableNames', false); 

% 参数
alphaVals = linspace(-1, 1, 41);
dcVals    = linspace(0, 5, 51);
nDc = length(dcVals);

% 选取 alpha = -1 
% 对应 Excel 的第 2 列 (第 1 列是 dc 标签)
targetAlphaIdx = 42; 

% 2. 提取数据
bestAI_vec = AI_grid(2:end, targetAlphaIdx); 
humanZero_vec = nan(nDc, 1);
matchDiff_vec = nan(nDc, 1);
levels = (0:5)';

for i = 1:nDc
    % 读取 X_grid 里的字符串，例如 "[0.100 0.200 0.300 0.200 0.100 0.100]"
    strVec = X_grid{i+1, targetAlphaIdx}{1};
    
    % 将字符串转为数值向量
    % 去掉方括号并解析
    cleanStr = strrep(strrep(strVec, '[', ''), ']', '');
    x = str2num(cleanStr)'; 
    
    if isempty(x), continue; end
    
    % 提取等级 0 占比
    humanZero_vec(i) = x(1);
    
    % 计算匹配等级差：当前分布 x 与当前最优 AI (bestAI_vec(i)) 的距离
    currentAI = bestAI_vec(i);
    matchDiff_vec(i) = sum(abs(levels - currentAI) .* x);
end

%% 3. 绘图：三位一体机制图
figure('Color', 'w', 'Position', [100, 100, 800, 500]);
hold on; grid on;

% --- 左轴：单调变化的社会状态 (等级 0 占比) ---
yyaxis left
h1 = plot(dcVals, humanZero_vec, '-r', 'LineWidth', 2.5); 
ylabel('Proportion of level-0 players', ...
       'FontSize', 12, 'Color', 'r');
set(gca, 'YColor', 'r');

% --- 右轴：最优策略与机制变量 ---
yyaxis right
h2 = plot(dcVals, bestAI_vec, '-b', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4);
h3 = plot(dcVals, matchDiff_vec, '--k', 'LineWidth', 1.5); % 匹配等级差

ylabel('Optimal AI level & Matching level gap', 'Interpreter', 'latex', 'FontSize', 12);
set(gca, 'YColor', 'k');

% 标注唯一的策略跳变点
switchIdx = find(diff(bestAI_vec) ~= 0, 1);
if ~isempty(switchIdx)
    xline(dcVals(switchIdx), ':m', 'LineWidth', 2, 'HandleVisibility', 'off');
    text(dcVals(switchIdx), max(bestAI_vec)*0.8, ' Shifting point', 'Color', 'm', 'FontWeight', 'bold');
end

xlabel('$d_c$: social tolerance threshold', 'Interpreter', 'latex');

title('$\alpha = -1$', ...
      'Interpreter', 'latex');

legend([h1, h2, h3], ...
    {'Human level-0 proportion', ...
     'Optimal AI level', ...
     'Matching level gap'}, ...
    'Location', 'northwest');

set(gca, 'FontSize', 11);
