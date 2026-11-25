%% 索引终极验证：分布向量比对
clear; clc;

% 1. 读取两个文件
X_all = readmatrix('X_fixedR03_allLevels.xlsx');
% 注意：读取 grid 文件时，我们需要解析里面的字符串
X_grid_table = readtable('X_fixedR03_grid.xlsx', 'ReadVariableNames', false); 
AI_grid = readmatrix('AI_bestLevel_grid.xlsx');

% 参数设置
alphaVals = linspace(-1, 1, 41);
dcVals    = linspace(0, 5, 51);

% 测试点：alpha = -1.0 (Idx 1), dc = 0.7 (Idx 8)
aIdx = 1; dIdx = 8;
targetAlpha = alphaVals(aIdx);
targetDc = dcVals(dIdx);

%% 2. 方式 A：从 AllLevels 提取 (使用我们最新的索引逻辑)
currRow = (aIdx-1) * 51 + dIdx;
AI_opt = AI_grid(dIdx+1, aIdx+1); 
cols_opt = (AI_opt * 6 + 1) : (AI_opt * 6 + 6);
x_from_all = X_all(currRow, cols_opt)';
x_from_all = x_from_all / sum(x_from_all); % 归一化

%% 3. 方式 B：从 Grid 字符串提取 (这是你原本认为的最优解)
% Grid 文件的索引：行 dIdx+1, 列 aIdx+1
str_vec = X_grid_table{dIdx+1, aIdx+1}{1};
% 解析字符串 "[0.000 0.123 ...]"
clean_str = strrep(strrep(str_vec, '[', ''), ']', '');
x_from_grid = str2num(clean_str)'; 

%% 4. 比对结果
diff_norm = norm(x_from_all - x_from_grid);
fprintf('--- 分布向量比对 (Alpha=%.1f, Dc=%.1f) ---\n', targetAlpha, targetDc);
fprintf('AllLevels 提取向量前三项: [%.4f, %.4f, %.4f]\n', x_from_all(1:3));
fprintf('Grid 字符串解析前三项:   [%.4f, %.4f, %.4f]\n', x_from_grid(1:3));
fprintf('向量欧氏距离 (Norm Diff): %.2e\n', diff_norm);

if diff_norm < 1e-4
    fprintf('状态: [绝对匹配] 索引提取逻辑无误！\n');
else
    fprintf('状态: [警告] 依然不匹配，请检查 alpha/dc 循环顺序。\n');
end