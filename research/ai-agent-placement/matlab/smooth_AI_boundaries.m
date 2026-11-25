function smooth_AI_boundaries(boundaryPoints)
% =====================================================================
% smooth_AI_boundaries.m —— polyfit 平滑边界（黑色线 + 无散点）
% =====================================================================

figure; hold on; box on;
set(gca,'FontSize',12);
xlabel('\alpha'); 
ylabel('$d_c$', 'Interpreter','latex');
title('AI Level Plot');

all_dc = [];   % 收集所有 dc 用于最终范围限制

for L1 = 0:5
    for L2 = (L1+1):5

        pts = boundaryPoints{L1+1, L2+1};
        if isempty(pts), continue; end

        dc    = pts(:,1);   % y
        alpha = pts(:,2);   % x

        all_dc = [all_dc; dc];

        % ====== 竖线边界 ======
        if all(abs(alpha) < 1e-12)
            plot(alpha, dc, 'k-', 'LineWidth', 2);   % 黑色线
            continue;
        end

        % ====== 非竖线：平滑拟合 ======
        [alpha, idx] = sort(alpha);
        dc = dc(idx);

        % 去重（保持你原本的逻辑）
        [alpha_unique, ~, ic] = unique(alpha);
        dc_unique = accumarray(ic, dc, [], @mean);

        x = alpha_unique;
        y = dc_unique;

        % 自动选择阶数（保留你的逻辑）
        if length(x) >= 5
            n = 3;
        else
            n = 2;
        end

        p = polyfit(x, y, n);

        xx = linspace(min(x), max(x), 300);
        yy = polyval(p, xx);

        % ====== 裁剪（保持你的封闭图效果） ======
        dc_min = min(all_dc);
        dc_max = max(all_dc);
        yy = max(min(yy, dc_max), dc_min);

        % ====== 绘图：全黑色曲线，无散点 ======
        plot(xx, yy, 'k-', 'LineWidth', 2);   % 黑色线
    end
end

% 最终 y 轴严格贴合（保持你原始版本）
ylim([min(all_dc), max(all_dc)]);

hold off;
end
