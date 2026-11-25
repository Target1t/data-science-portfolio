function dbar_table = compute_dbar_compare()
% ============================================================
% 计算：有 AI 与 无 AI 的期望等级差 dbar（按 AI 最优等级分组）
%
% 输出：6×2 matrix
%   col1 = 有 AI（r=0.3）
%   col2 = 无 AI（r=0）
%
% 使用:
%   dbar_table = compute_dbar_compare()
%
% ============================================================

    % ===== 固定参数 =====
    r_AI  = 0.3;      % 有 AI
    r_noAI = 0.0;     % 无 AI
    epsilon = 0.02;
    p0 = 0;
    K = 5;            % active 1..5

    % ===== 读取文件 =====
    disp('读取 Excel 文件（有 AI）...');
    X_AI = readcell('X_fixedR03_grid.xlsx');
    AIdata = readcell('AI_bestLevel_grid.xlsx');

    disp('读取 Excel 文件（无 AI）...');
    X_noAI = readcell('X_fixedR0_grid_smooth.xlsx');  % 无AI时的人类最优分布

    % 去掉行列标签
    X_AI_mat = X_AI(2:end, 2:end);
    X_noAI_mat = X_noAI(2:end, 2:end);
    AImat = AIdata(2:end, 2:end);

    nDC = size(X_AI_mat,1);
    nAlpha = size(X_AI_mat,2);

    % ===== 预分配 =====
    dbar_AI  = nan(nDC, nAlpha);
    dbar_noAI = nan(nDC, nAlpha);

    % ============================================================
    %% 计算有 AI 的 dbar（r=0.3）
    % ============================================================

    disp('计算 有 AI 的 dbar (r=0.3) ...');

    for i = 1:nDC
        for j = 1:nAlpha

            %% ---- 读取人类 x (有 AI 情况) ----
            xstr = X_AI_mat{i,j};
            if ~ischar(xstr), continue; end
            if strcmp(xstr,'NaN'), continue; end
            x = sscanf(xstr,'[%f %f %f %f %f %f]');
            if isempty(x), continue; end

            %% ---- 读取最优 AI 等级 ----
            AIL = AImat{i,j};
            if ischar(AIL)
                if strcmp(AIL,'NaN'), continue; end
                AIL = str2double(AIL);
            end
            if isnan(AIL), continue; end

            %% ---- y_fixed ----
            y_fixed = zeros(6,1);
            y_fixed(AIL+1) = 1;

            %% ---- active 权重（有AI）----
            x_act = x(2:6);
            y_act = y_fixed(2:6);

            active_eff = sum(x_act) + r_AI * sum(y_act);
            if active_eff < 1e-12
                wgt = ones(K,1)/K;
            else
                wgt = (x_act + r_AI * y_act) / active_eff;
            end

            %% ---- 针对每个等级 h 计算期望差 ----
            E_h = zeros(K,1);
            for h = 1:K
                dvec = abs((1:K) - h);
                E_h(h) = sum(wgt(:)' .* dvec);
            end

            %% ---- 针对 system 的加权 ----
            weight_h = x_act / sum(x_act);
            dbar_AI(i,j) = sum(weight_h(:)' .* E_h(:)');

        end
    end


    % ============================================================
    %% 计算 无 AI 的 dbar（r=0）
    % ============================================================

    disp('计算 无 AI 的 dbar (r=0) ...');

    for i = 1:nDC
        for j = 1:nAlpha

            %% ---- 读取人类 x (无 AI 情况) ----
            xstr = X_noAI_mat{i,j};
            if ~ischar(xstr), continue; end
            if strcmp(xstr,'NaN'), continue; end
            x = sscanf(xstr,'[%f %f %f %f %f %f]');
            if isempty(x), continue; end

            %% ---- y_fixed = 0（无 AI） ----
            y_act = zeros(K,1);

            %% ---- active 权重（无AI r=0）----
            x_act = x(2:6);
            active_eff = sum(x_act);  % no AI contribution

            if active_eff < 1e-12
                wgt = ones(K,1)/K;
            else
                wgt = x_act / active_eff;
            end

            %% ---- 针对每个等级 h 计算期望差 ----
            E_h = zeros(K,1);
            for h = 1:K
                dvec = abs((1:K) - h);
                E_h(h) = sum(wgt(:)' .* dvec);
            end

            %% ---- 针对 system 的加权 ----
            weight_h = x_act / sum(x_act);
            dbar_noAI(i,j) = sum(weight_h(:)' .* E_h(:)');

        end
    end

    % ============================================================
    %% 按 AI 最优等级分组求平均
    % ============================================================

    disp('计算分组平均 (AI=0..5) ...');

    % 先将 AImat 转成数值矩阵
    AInum = nan(nDC, nAlpha);
    for i = 1:nDC
        for j = 1:nAlpha
            aij = AImat{i,j};
            if ischar(aij)
                if strcmp(aij,'NaN'), AInum(i,j)=nan;
                else, AInum(i,j)=str2double(aij);
                end
            else
                AInum(i,j) = aij;
            end
        end
    end

    dbar_by_AI = nan(6,1);
    dbar_by_noAI = nan(6,1);

    for k = 0:5
        mask = (AInum == k);

        vals_AI   = dbar_AI(mask);
        vals_noAI = dbar_noAI(mask);

        dbar_by_AI(k+1)   = mean(vals_AI(~isnan(vals_AI)));
        dbar_by_noAI(k+1) = mean(vals_noAI(~isnan(vals_noAI)));
    end

    % ===== 输出最终表格 =====
    dbar_table = [dbar_by_AI, dbar_by_noAI];

    disp('====== dbar 对比表（AI=0..5）======');
    disp('  col1 = 有AI(r=0.3), col2 = 无AI(r=0)');
    disp(dbar_table);

end
