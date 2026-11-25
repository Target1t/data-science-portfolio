 function plot_distribution_vs_dc_clustered()

    % ====== 读取 Excel ======
    X_data = readcell('X_fixedR03_grid.xlsx');
    X_mat  = X_data(2:end,2:end);

    dcVals    = cellfun(@str2double, X_data(2:end,1));
    alphaVals = cellfun(@str2double, X_data(1,2:end));

    [~, idxNeg] = min(abs(alphaVals - (-1)));
    [~, idxPos] = min(abs(alphaVals - (+1)));

    plot_single_alpha_clustered(dcVals, X_mat(:,idxNeg), -1);
    plot_single_alpha_clustered(dcVals, X_mat(:,idxPos), +1);

end


%% ====== 单个 alpha 下的聚类分布 ======
function plot_single_alpha_clustered(dcVals, col_X, alpha_value)

    n = length(dcVals);
    X = nan(n,6);

    for i = 1:n
        if ischar(col_X{i}) && ~strcmp(col_X{i},'NaN')
            v = sscanf(col_X{i}, '[%f %f %f %f %f %f]');
            if numel(v)==6
                X(i,:) = v';
            end
        end
    end

    low  = X(:,1) + X(:,2);
    mid  = X(:,3) + X(:,4);
    high = X(:,5) + X(:,6);

    % ===== 绘图 =====
    figure; hold on; box on;

    % --- 离散跳变竖线 ---
    if alpha_value == -1
        xline(0.7,'--k','LineWidth',1.5);
    elseif alpha_value == 1
        xline(2.1,'--k','LineWidth',1.5);
    end

    plot(dcVals, low,  'LineWidth',2);
    plot(dcVals, mid,  'LineWidth',2);
    plot(dcVals, high, 'LineWidth',2);

    xlabel('$d_c$','Interpreter','latex');
    ylabel('$x$','Interpreter','latex');

    legend({'low (0–1)','mid (2–3)','high (4–5)'}, ...
           'Interpreter','latex','Location','best');

    title(['$\alpha = ', num2str(alpha_value),'$'], ...
          'Interpreter','latex');
end
