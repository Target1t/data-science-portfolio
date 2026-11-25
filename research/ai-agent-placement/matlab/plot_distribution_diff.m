function plot_distribution_diff()

    % ===== 读取 Excel =====
    X0_data = readcell('X_fixedR0_grid_smooth.xlsx');   % r = 0
    X1_data = readcell('X_fixedR03_grid.xlsx');        % r = 0.3

    X0 = X0_data(2:end,2:end);   % 51 × 41
    X1 = X1_data(2:end,2:end);   % 51 × 41

    % 读取 dc 与 alpha
    dcVals    = cellfun(@str2double, X0_data(2:end,1));   % 51
    alphaVals = cellfun(@str2double, X0_data(1,2:end));   % 41

    % ===== 找 alpha = -1 和 alpha = +1 =====
    [~, idxNeg] = min(abs(alphaVals - (-1)));
    [~, idxPos] = min(abs(alphaVals - (+1)));

    % ===== 各画一张图 =====
    plot_single_alpha_diff(dcVals, X0(:,idxNeg), X1(:,idxNeg), -1);
    plot_single_alpha_diff(dcVals, X0(:,idxPos), X1(:,idxPos), +1);

end


%% ========== 画 Δx_h(dc) = X_r0.3 − X_r0 =============
function plot_single_alpha_diff(dcVals, col0, col1, alpha_value)

    n = length(dcVals);
    X0 = nan(n,6);
    X1 = nan(n,6);

    % ---- 读取两个分布 ----
    for i = 1:n
        s0 = col0{i};
        s1 = col1{i};

        if ischar(s0) && ~strcmp(s0,'NaN')
            x0 = sscanf(s0,'[%f %f %f %f %f %f]');
            if length(x0)==6
                X0(i,:) = x0';
            end
        end

        if ischar(s1) && ~strcmp(s1,'NaN')
            x1 = sscanf(s1,'[%f %f %f %f %f %f]');
            if length(x1)==6
                X1(i,:) = x1';
            end
        end
    end

    % ---- 差分 ----
    DX = X1 - X0;

    % ===== 图 =====
    figure; hold on; box on;

    % ===== AI 等级离散跳变位置（竖虚线）=====
    if alpha_value == -1
        % dc = 0.7 -> 0.8 发生跳变
        xline(0.7, '--k', 'LineWidth', 1.5);
    elseif alpha_value == 1
        % dc = 2.1 -> 2.2 发生跳变
        xline(2.1, '--k', 'LineWidth', 1.5);
    end

    colors = lines(6);
    lw = 2;

    for h = 1:6
        plot(dcVals, DX(:,h), 'LineWidth', lw, 'Color', colors(h,:));
    end

    xlabel('$d_c$', 'Interpreter','latex');
    ylabel('$\Delta x_h$', 'Interpreter','latex');

    legend({'level 0','level 1','level 2','level 3','level 4','level 5'}, ...
        'Interpreter','latex','Location','best');

    title(['$\alpha = ', num2str(alpha_value),'$'], ...
        'Interpreter','latex');

end
