function plot_mechanism_analysis()

    % ========= 读取已有 Excel =========
    X_noAI_data  = readcell('X_fixedR0_grid_smooth.xlsx');
    X_AI_data    = readcell('X_fixedR03_grid.xlsx');
    AI_data      = readcell('AI_bestLevel_grid.xlsx');

    % 去掉标题行列
    X_noAI = X_noAI_data(2:end,2:end);   % 51 × 41
    X_AI   = X_AI_data(2:end,2:end);
    AImat  = AI_data(2:end,2:end);

    % dc, alpha 轴
    dcVals    = cellfun(@str2double, X_noAI_data(2:end,1));   % 51
    alphaVals = cellfun(@str2double, X_noAI_data(1,2:end));   % 41

    % ========= 找 alpha = -1 与 alpha = +1 的列 =========
    [~, idxAlphaNeg] = min(abs(alphaVals - (-1)));
    [~, idxAlphaPos] = min(abs(alphaVals - (+1)));

    % ========= 绘图（alpha = -1）=========
    plot_single_alpha(dcVals, ...
        X_noAI(:, idxAlphaNeg), ...
        X_AI(:, idxAlphaNeg), ...
        AImat(:, idxAlphaNeg), ...
        -1);

    % ========= 绘图（alpha = +1）=========
    plot_single_alpha(dcVals, ...
        X_noAI(:, idxAlphaPos), ...
        X_AI(:, idxAlphaPos), ...
        AImat(:, idxAlphaPos), ...
        +1);

end


%% ========== 对某个 alpha 画一张机制分析图 ==========
function plot_single_alpha(dcVals, col_noAI, col_AI, col_AIL, alpha_value)

    n = length(dcVals);

    d1 = nan(n,1);   % within no-AI
    d2 = nan(n,1);   % within AI
    d3 = nan(n,1);   % between AI

    r = 0.3;

    for i = 1:n

        % --- X (no AI) ---
        str0 = col_noAI{i};
        if ischar(str0) && ~strcmp(str0,'NaN')
            x0 = sscanf(str0,'[%f %f %f %f %f %f]');
            if ~isempty(x0)
                d1(i) = compute_within(x0);
            end
        end

        % --- X (AI) ---
        str1 = col_AI{i};
        strA = col_AIL{i};

        if ischar(str1) && ~strcmp(str1,'NaN')
            x1 = sscanf(str1,'[%f %f %f %f %f %f]');
        else
            x1 = [];
        end

        if ischar(strA)
            if strcmp(strA,'NaN'), AIL = nan; 
            else AIL = str2double(strA);
            end
        else
            AIL = strA;
        end

        if ~isempty(x1) && ~isnan(AIL)
            d2(i) = compute_within(x1);                % within AI
            d3(i) = compute_between(x1, AIL, r);       % between AI
        end
    end

    % ========= 画图 =========
    figure; hold on; box on;

    plot(dcVals, d1, 'LineWidth',2);
    plot(dcVals, d2, 'LineWidth',2);
    plot(dcVals, d3, 'LineWidth',2);

    legend({'within-noAI','within-AI','between-AI'}, ...
           'Location','best','Interpreter','latex');

    xlabel('$d_c$','Interpreter','latex');
    ylabel('$\bar{d}$','Interpreter','latex');

    title(['$\alpha = ', num2str(alpha_value),'$'], 'Interpreter','latex');

end


%% ===== 下面是你的 compute_within 和 compute_between，原样复制 =====

function d_within = compute_within(x)
    x_act = x(2:6);
    S = sum(x_act);
    if S < 1e-12
        d_within = NaN; return;
    end

    w_h = x_act / S;
    K = 5;
    E_h = zeros(K,1);

    for h = 1:K
        dvec = abs((1:K) - h);
        wjt  = x_act / S;
        E_h(h) = sum(wjt(:)' .* dvec);
    end

    d_within = sum(w_h(:)' .* E_h(:)');
end


function d_between = compute_between(x, AIL, r)
    x_act = x(2:6);
    S = sum(x_act);

    active_eff = S + r;
    if active_eff < 1e-12
        d_between = NaN; return;
    end

    P = r / active_eff;
    w_h = x_act / S;

    L = AIL;
    K = 5;

    d_h = abs((1:K)' - L);

    d_between = P * sum(w_h(:)' .* d_h(:)');
end
