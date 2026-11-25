function plot_distribution_diff_clustered()

    % ===== 读取 Excel =====
    X0_data = readcell('X_fixedR0_grid_smooth.xlsx');   % r = 0
    X1_data = readcell('X_fixedR03_grid.xlsx');        % r = 0.3

    X0 = X0_data(2:end,2:end);   % 51 × 41
    X1 = X1_data(2:end,2:end);   % 51 × 41

    % 读取 dc 与 alpha
    dcVals    = cellfun(@str2double, X0_data(2:end,1));
    alphaVals = cellfun(@str2double, X0_data(1,2:end));

    % ===== 找 alpha = -1 和 +1 =====
    [~, idxNeg] = min(abs(alphaVals - (-1)));
    [~, idxPos] = min(abs(alphaVals - (+1)));

    % ===== 各画一张图 =====
    plot_single_alpha_diff_clustered(dcVals, X0(:,idxNeg), X1(:,idxNeg), -1);
    plot_single_alpha_diff_clustered(dcVals, X0(:,idxPos), X1(:,idxPos), +1);

end


%% ====== Δx 的等级聚类展示 ======
function plot_single_alpha_diff_clustered(dcVals, col0, col1, alpha_value)

    n = length(dcVals);
    X0 = nan(n,6);
    X1 = nan(n,6);

    % ---- 读取分布 ----
    for i = 1:n
        if ischar(col0{i}) && ~strcmp(col0{i},'NaN')
            v0 = sscanf(col0{i}, '[%f %f %f %f %f %f]');
            if numel(v0)==6, X0(i,:) = v0'; end
        end
        if ischar(col1{i}) && ~strcmp(col1{i},'NaN')
            v1 = sscanf(col1{i}, '[%f %f %f %f %f %f]');
            if numel(v1)==6, X1(i,:) = v1'; end
        end
    end

    % ---- 聚类 ----
    low0  = X0(:,1) + X0(:,2);
    mid0  = X0(:,3) + X0(:,4);
    high0 = X0(:,5) + X0(:,6);

    low1  = X1(:,1) + X1(:,2);
    mid1  = X1(:,3) + X1(:,4);
    high1 = X1(:,5) + X1(:,6);

    Dlow  = low1  - low0;
    Dmid  = mid1  - mid0;
    Dhigh = high1 - high0;

    % ===== 绘图 =====
    figure; hold on; box on;

    % --- 离散跳变竖线 ---
    if alpha_value == -1
        xline(0.7,'--k','LineWidth',1.5);
    elseif alpha_value == 1
        xline(2.1,'--k','LineWidth',1.5);
    end

    plot(dcVals, Dlow,  'LineWidth',2);
    plot(dcVals, Dmid,  'LineWidth',2);
    plot(dcVals, Dhigh, 'LineWidth',2);

    xlabel('$d_c$','Interpreter','latex');
    ylabel('$\Delta x$','Interpreter','latex');

    legend({'low (0–1)','mid (2–3)','high (4–5)'}, ...
           'Interpreter','latex','Location','best');

    title(['$\alpha = ', num2str(alpha_value),'$'], ...
          'Interpreter','latex');
end
