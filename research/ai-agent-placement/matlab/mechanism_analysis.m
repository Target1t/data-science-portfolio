function mechanism_analysis()
% ============================================================
% 可调机制分析（固定 alpha，任意两个 AI 等级对比）
%
% 名字：mechanism_analysis  —— 不允许更改
% ============================================================

    %% ===================== 用户设置 ==========================
    alpha_target = -1;      % 你想分析哪个 α，例如 -1, 0, 0.5, 1 ...
    AI_level_A   = 1;       % 要比较的第一个 AI 等级（0~5）
    AI_level_B   = 5;       % 要比较的第二个 AI 等级（0~5）
    % ===========================================================

    %% ====== 读取保存的数据 ======
    EL = readmatrix('EL_fixedR03_allLevels.xlsx');  % 2091×6

    dcVals    = linspace(0,5,51);
    alphaVals = linspace(-1,1,41);

    %% ====== reshape 成 51 × 41 × 6 ======
    EL_grid = nan(51,41,6);
    idx = 1;
    for j = 1:41
        for i = 1:51
            EL_grid(i,j,:) = EL(idx,:);
            idx = idx + 1;
        end
    end

    %% ====== 找到最接近 alpha_target 的列 ======
    [~, jstar] = min(abs(alphaVals - alpha_target));

    %% ====== 提取曲线 ======
    dc = dcVals(:);
    EL_A = EL_grid(:, jstar, AI_level_A+1);
    EL_B = EL_grid(:, jstar, AI_level_B+1);

    %% ====== 绘图 ======
    figure; hold on; box on;

    set(gca,'FontSize',16);
    xlabel('$d_c$', 'Interpreter','latex');
    ylabel('$E[L]$', 'Interpreter','latex');

    % 标题自动生成
    title(sprintf('$\\alpha=%g,\\;E[L]\\;\\mathrm{for}\\;AI=%d\\;\\mathrm{and}\\;AI=%d$', ...
              alpha_target, AI_level_A, AI_level_B), ...
              'Interpreter','latex');


    % 论文风格：黑实线 + 黑虚线
    plot(dc, EL_A, 'k-',  'LineWidth', 2);
    plot(dc, EL_B, 'k--', 'LineWidth', 2);

    legend({sprintf('AI=%d',AI_level_A), sprintf('AI=%d',AI_level_B)}, ...
           'Location','best', ...
           'Interpreter','latex');

    hold off;

end
