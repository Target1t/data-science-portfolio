function boundaryPoints = get_AI_boundaries(filename)
% ============================================================
%  get_AI_boundaries.m   —— 增强版
%
%  功能：
%    1. 从 AI_bestLevel_grid.xlsx 中读取相图；
%    2. 自动识别不同 AI 等级之间的所有边界；
%    3. 在命令行直接输出：「出现哪些边界」以及「每条边界的全部坐标」；
%    4. 返回 boundaryPoints{L1+1, L2+1} = [dc, alpha] 二维坐标列表。
%
%  输入：
%        filename  ——  AI_bestLevel_grid.xlsx
%
%  注意：
%        Excel 第一列 = dc；第一行 = alpha；
%        中间矩阵 = AI_bestLevel (0–5)
% ============================================================

%% ===== 读取 Excel =====
raw = readcell(filename);

alphaVals = cellfun(@str2double, raw(1,2:end));   % 第一行 alpha
dcVals    = cellfun(@str2double, raw(2:end,1));   % 第一列 dc

AI = raw(2:end, 2:end);
AI = cellfun(@str2double, AI);                    % 转 double

[nDC, nAlpha] = size(AI);

%% ===== 初始化边界存储 =====
boundaryPoints = cell(6,6);  % AI 等级 0~5 → 1~6 索引

%% ===== 查找边界 =====
for i = 1:nDC
    for j = 1:nAlpha

        L = AI(i,j);
        if isnan(L), continue; end

        % 四邻域
        neighbors = [];
        if i>1,     neighbors = [neighbors; AI(i-1,j)]; end
        if i<nDC,   neighbors = [neighbors; AI(i+1,j)]; end
        if j>1,     neighbors = [neighbors; AI(i,j-1)]; end
        if j<nAlpha,neighbors = [neighbors; AI(i,j+1)]; end

        neighbors = unique(neighbors(~isnan(neighbors)));

        for Ln = neighbors'
            if Ln ~= L
                dc    = dcVals(i);
                alpha = alphaVals(j);

                boundaryPoints{L+1, Ln+1} = [
                    boundaryPoints{L+1, Ln+1};
                    dc, alpha
                ];
            end
        end
    end
end

%% ===== 整理输出到命令行 =====
fprintf('\n=============================================\n');
fprintf(' AI 等级边界识别完毕，下面是检测到的全部边界：\n');
fprintf('=============================================\n\n');

for L1 = 0:5
    for L2 = (L1+1):5   % 防止重复（0-1 与 1-0 只输出一次）
        pts = boundaryPoints{L1+1, L2+1};
        if ~isempty(pts)
            fprintf('🔥 边界：AI = %d  <-->  AI = %d\n', L1, L2);
            fprintf('   边界点数量：%d\n', size(pts,1));
            fprintf('   坐标列表：(dc , alpha)\n');

            for k = 1:size(pts,1)
                fprintf('     (%.4f , %.4f)\n', pts(k,1), pts(k,2));
            end

            fprintf('\n');
        end
    end
end

fprintf('=============================================\n');
fprintf('  全部边界已输出完毕。\n');
fprintf('=============================================\n\n');
end

