%% ============================================================
%  完整平均等级差（含人类–AI 匹配）
%  固定 alpha = -1 / 1
%  横轴 dc
%  六条线：假设 AI 选择等级 k = 0..5
%% ============================================================

clear; clc;

%% ===== 参数（必须一致）=====
alphaVals = linspace(-1, 1, 41);
dcVals    = linspace(0, 5, 51);
r = 0.3;

nAlpha = length(alphaVals);
nDc    = length(dcVals);

%% ===== 读取 Excel =====
raw = readmatrix('X_fixedR03_allLevels.xlsx');

% reshape: (dc, alpha, human_level, AI_level)
Xall = reshape(raw, nDc, nAlpha, 6, 6);

%% ===== 分析 alpha =====
alphas_to_plot = [-1, 1];
colors = lines(6);

for aidx = 1:length(alphas_to_plot)

    alpha_target = alphas_to_plot(aidx);
    [~, j_alpha] = min(abs(alphaVals - alpha_target));

    meanGap_full = nan(nDc, 6);  % dc × AI_level

    %% ===== 主循环 =====
    for i = 1:nDc
        for k = 0:5

            x = squeeze(Xall(i, j_alpha, :, k+1));

            if any(isnan(x)) || abs(sum(x)-1) > 1e-6
                continue;
            end

            % AI 分布（单等级）
            y = zeros(6,1);
            y(k+1) = 1;

            % 有效匹配对象分布
            p_eff = (x + r*y) / (1 + r);

            % ===== 计算完整平均等级差 =====
            gap = 0;
            for ii = 0:5
                for jj = 0:5
                    gap = gap + x(ii+1) * p_eff(jj+1) * abs(ii - jj);
                end
            end

            meanGap_full(i, k+1) = gap;
        end
    end

    %% ===== 绘图 =====
    figure; hold on;

    for k = 0:5
        plot(dcVals, meanGap_full(:,k+1), ...
            'LineWidth', 2, ...
            'Color', colors(k+1,:), ...
            'DisplayName', sprintf('AI level %d', k));
    end

    xlabel('$d_c$','Interpreter','latex');
    ylabel('Mean level difference (human + AI)','Interpreter','latex');
    title(sprintf('Mean level gap incl. AI (\\alpha = %.1f)', alphaVals(j_alpha)), ...
        'Interpreter','latex');

    legend('Location','best');
    grid on;
end
