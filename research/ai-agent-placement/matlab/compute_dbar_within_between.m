function dbar_table = compute_dbar_within_between()
% =============================================================
%   计算三类等级差：
%     1) 无AI时（r=0）的组内等级差 d_within_noAI
%     2) 有AI时（r=0.3）的组内等级差 d_within_AI
%     3) 有AI时（r=0.3）的组间等级差 d_between_AI
%
%   输出：6×3 的矩阵（按最优 AI 等级 0..5 分组）
%
%   使用文件：
%     X_fixedR0_grid_smooth.xlsx   （人类最优分布，无AI）
%     X_fixedR03_grid.xlsx         （人类最优分布，有AI）
%     AI_bestLevel_grid.xlsx       （AI 最优等级）
% =============================================================

    % ---------- 固定参数 ----------
    r = 0.3;           % AI 参与比例
    K = 5;             % active levels 1..5 (index 2..6)

    % ---------- 读取 Excel ----------
    disp('读取 Excel 文件...');
    X_noAI_data  = readcell('X_fixedR0_grid_smooth.xlsx');
    X_AI_data    = readcell('X_fixedR03_grid.xlsx');
    AI_data      = readcell('AI_bestLevel_grid.xlsx');

    X_noAI  = X_noAI_data(2:end, 2:end);   % 51×41
    X_AI    = X_AI_data(2:end, 2:end);
    AImat   = AI_data(2:end, 2:end);

    nDC    = size(X_AI,1);   % 51
    nAlpha = size(X_AI,2);   % 41

    % ---------- 存储三个 d̄ ----------
    d_within_noAI = nan(nDC,nAlpha);
    d_within_AI   = nan(nDC,nAlpha);
    d_between_AI  = nan(nDC,nAlpha);

    disp('逐点计算 d̄ ...');

    for i = 1:nDC
        for j = 1:nAlpha

            % =====================================================
            % 1. 无 AI 情况（r=0，只有组内）
            % =====================================================
            xstr0 = X_noAI{i,j};
            if ischar(xstr0)
                if strcmp(xstr0,'NaN'), continue; end
                x0 = sscanf(xstr0,'[%f %f %f %f %f %f]');
            else
                continue;
            end
            if isempty(x0), continue; end

            d_within_noAI(i,j) = compute_within(x0);

            % =====================================================
            % 2. 有 AI 情况：读取人类最优分布
            % =====================================================
            xstr = X_AI{i,j};
            AIL  = AImat{i,j};

            if ischar(xstr)
                if strcmp(xstr,'NaN'), continue; end
                x = sscanf(xstr,'[%f %f %f %f %f %f]');
            else
                continue;
            end
            if isempty(x), continue; end

            if ischar(AIL)
                if strcmp(AIL,'NaN'), continue; end
                AIL = str2double(AIL);
            end
            if isnan(AIL), continue; end

            % =====================================================
            % 3. 有 AI 时的组内 d̄
            % =====================================================
            d_within_AI(i,j) = compute_within(x);

            % =====================================================
            % 4. 有 AI 时的组间 d̄
            % =====================================================
            d_between_AI(i,j) = compute_between(x, AIL, r);

        end
    end

    % ---------- 读取 AI 最优等级（数值化） ----------
    AInum = convert_AI(AImat);

    % ---------- 按 AI 最优等级分组求平均 ----------
    dbar_table = nan(6,3);   % AI=0..5 × (within_noAI, within_AI, between_AI)

    disp('按最优 AI 等级分组求均值 ...');

    for k = 0:5
        mask = (AInum == k);

        vals1 = d_within_noAI(mask);
        vals2 = d_within_AI(mask);
        vals3 = d_between_AI(mask);

        dbar_table(k+1,1) = mean(vals1(~isnan(vals1)));
        dbar_table(k+1,2) = mean(vals2(~isnan(vals2)));
        dbar_table(k+1,3) = mean(vals3(~isnan(vals3)));
    end

    % ---------- 输出 ----------
    disp('====== 最终三列 d̄（AI=0..5）======');
    disp(array2table(dbar_table, ...
        'VariableNames', {'within_noAI','within_AI','between_AI'}));

end


%% =============================================================
%% 计算组内等级差（无AI or 有AI）
%% =============================================================
function d_within = compute_within(x)
    x_act = x(2:6);
    S = sum(x_act);
    if S < 1e-12
        d_within = NaN;
        return;
    end

    w_h = x_act / S;   % 每个等级 h 在参与者中的占比

    K = 5;
    E_h = zeros(K,1);

    % 对所有 h=1..5 的玩家
    for h = 1:K
        dvec = abs((1:K) - h);                 % |h-j|
        wjt  = x_act / S;                      % 人类匹配权重
        E_h(h) = sum(wjt(:)' .* dvec);
    end

    % 系统加权
    d_within = sum(w_h(:)' .* E_h(:)');
end


%% =============================================================
%% 计算组间等级差（人类匹配 AI）
%% =============================================================
function d_between = compute_between(x, AIL, r)

    x_act = x(2:6);
    S = sum(x_act);

    active_eff = S + r;
    if active_eff < 1e-12
        d_between = NaN;
        return;
    end

    % 匹配AI的概率
    P = r / active_eff;

    % 玩家在参与者中的比重
    w_h = x_act / S;

    % AI 等级
    L = AIL;

    K = 5;
    d_h = zeros(K,1);

    % 每个等级 h 与 AI 的等级差
    for h = 1:K
        d_h(h) = abs(h - L);
    end

    % 组间等级差
    d_between = P * sum(w_h(:)' .* d_h(:)');
end


%% =============================================================
%% 将 AI bestLevel 从 cell 转成 double
%% =============================================================
function AInum = convert_AI(AImat)
    [n,m] = size(AImat);
    AInum = nan(n,m);

    for i = 1:n
        for j = 1:m
            val = AImat{i,j};
            if ischar(val)
                if strcmp(val,'NaN')
                    AInum(i,j) = nan;
                else
                    AInum(i,j) = str2double(val);
                end
            else
                AInum(i,j) = val;
            end
        end
    end
end
