%% ============================================================
%  人类平均等级差分析（反事实）
%  固定 alpha = -1 / 1
%  横轴 dc
%  六条线：假设 AI 选择等级 k = 0..5
%% ============================================================

clear; clc;

%% ===== 基本参数（必须与你原始程序一致）=====
alphaVals = linspace(-1, 1, 41);
dcVals    = linspace(0, 5, 51);

nAlpha = length(alphaVals);
nDc    = length(dcVals);

%% ===== 读取 Excel =====
raw = readmatrix('X_fixedR03_allLevels.xlsx');

% reshape 回 (dc, alpha, human_level, AI_level)
Xall = reshape(raw, nDc, nAlpha, 6, 6);

%% ===== 要分析的 alpha =====
alphas_to_plot = [-1, 1];
colors = lines(6);

for aidx = 1:length(alphas_to_plot)

    alpha_target = alphas_to_plot(aidx);
    [~, j_alpha] = min(abs(alphaVals - alpha_target));

    % 存储：dc × AI_level
    meanGap = nan(nDc, 6);

    %% ===== 遍历 dc 和 AI 等级 =====
    for i = 1:nDc
        for k = 0:5

            x = squeeze(Xall(i, j_alpha, :, k+1));

            if any(isnan(x)) || abs(sum(x)-1) > 1e-6
                continue;
            end

            % ===== 计算平均等级差 =====
            gap = 0;
            for ii = 0:5
                for jj = 0:5
                    gap = gap + x(ii+1) * x(jj+1) * abs(ii - jj);
                end
            end

            meanGap(i, k+1) = gap;
        end
    end

    %% ===== 绘图 =====
    figure; hold on;

    for k = 0:5
        plot(dcVals, meanGap(:,k+1), ...
            'LineWidth', 2, ...
            'Color', colors(k+1,:), ...
            'DisplayName', sprintf('AI level %d', k));
    end

    xlabel('$d_c$','Interpreter','latex');
    ylabel('Mean level difference','Interpreter','latex');
    title(sprintf('Mean human level gap (\\alpha = %.1f)', alphaVals(j_alpha)), ...
        'Interpreter','latex');

    legend('Location','best');
    grid on;
end
