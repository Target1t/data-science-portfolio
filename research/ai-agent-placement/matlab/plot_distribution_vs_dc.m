function plot_distribution_vs_dc()

    % ====== 读取 Excel ======
    X_data = readcell('X_fixedR03_grid.xlsx');    % 有AI时的分布
    X_mat  = X_data(2:end,2:end);                 % 51 × 41

    % 读取 dc / alpha 轴
    dcVals    = cellfun(@str2double, X_data(2:end,1));   % 51
    alphaVals = cellfun(@str2double, X_data(1,2:end));   % 41

    % ====== 找 alpha = -1 和 +1 对应的列 ======
    [~, idxAlphaNeg] = min(abs(alphaVals - (-1)));
    [~, idxAlphaPos] = min(abs(alphaVals - (+1)));

    % ====== 绘图：alpha=-1 ======
    plot_single_alpha_distribution(dcVals, X_mat(:,idxAlphaNeg), -1);

    % ====== 绘图：alpha=+1 ======
    plot_single_alpha_distribution(dcVals, X_mat(:,idxAlphaPos), +1);

end


%% ================= 画单个 alpha 下的分布变化（6 条线） =================
function plot_single_alpha_distribution(dcVals, col_X, alpha_value)

    n = length(dcVals);

    % 6 个等级的分布轨迹
    X = nan(n,6);

    for i = 1:n
        str = col_X{i};
        if ischar(str) && ~strcmp(str,'NaN')
            xvec = sscanf(str,'[%f %f %f %f %f %f]');
            if length(xvec)==6
                X(i,:) = xvec(:)';
            end
        end
    end

    % ===== 绘图 =====
    figure; hold on; box on;

    % ===== AI 等级离散跳变位置（竖虚线）=====
    if alpha_value == -1
        % dc = 0.7 -> 0.8 发生跳变
        xline(0.7, '--k', 'LineWidth', 1.5);
    elseif alpha_value == 1
        % dc = 2.1 -> 2.2 发生跳变
        xline(2.1, '--k', 'LineWidth', 1.5);
    end

    lw = 2;
    plot(dcVals, X(:,1), 'LineWidth',lw);
    plot(dcVals, X(:,2), 'LineWidth',lw);
    plot(dcVals, X(:,3), 'LineWidth',lw);
    plot(dcVals, X(:,4), 'LineWidth',lw);
    plot(dcVals, X(:,5), 'LineWidth',lw);
    plot(dcVals, X(:,6), 'LineWidth',lw);

    xlabel('$d_c$','Interpreter','latex');
    ylabel('$x_h$','Interpreter','latex');

    legend({'level 0','level 1','level 2','level 3','level 4','level 5'},...
           'Interpreter','latex','Location','best');

    title(['$\alpha = ', num2str(alpha_value), '$'], ...
          'Interpreter','latex');

end
