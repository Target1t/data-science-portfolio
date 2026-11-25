function U_diff_table = compute_utility_change()
% =====================================================================
% 计算等级效用变化（有AI − 无AI），并按 AI 最优等级聚合求平均
% =====================================================================

    disp('读取 Excel 文件...');

    %% ====== 读取 Excel ======
    X_noAI_data = readcell('X_fixedR0_grid_smooth.xlsx');
    X_AI_data   = readcell('X_fixedR03_grid.xlsx');
    AI_data     = readcell('AI_bestLevel_grid.xlsx');

    X_noAI = X_noAI_data(2:end, 2:end);
    X_AI   = X_AI_data(2:end, 2:end);
    AImat  = AI_data(2:end, 2:end);

    [nDC, nAlpha] = size(X_AI);   % 51 × 41 = 2091 点

    %% ====== 存储效用向量 ======
    U_noAI = nan(nDC, nAlpha, 6);
    U_AI   = nan(nDC, nAlpha, 6);

    disp('逐点计算效用分量 U_k = k * x_k ...');

    for i = 1:nDC
        for j = 1:nAlpha

            % ========= 无 AI =========
            s0 = X_noAI{i,j};
            if ischar(s0) && ~strcmp(s0,"NaN")
                x0 = sscanf(s0,"[%f %f %f %f %f %f]");
                U_noAI(i,j,:) = (0:5).*x0';
            end

            % ========= 有 AI =========
            sA = X_AI{i,j};
            if ischar(sA) && ~strcmp(sA,"NaN")
                xA = sscanf(sA,"[%f %f %f %f %f %f]");
                U_AI(i,j,:) = (0:5).*xA';
            end

        end
    end

    %% ====== 转换 AI 最优等级矩阵 ======
    AInum = convert_AI(AImat);

    %% ====== reshape 成二维矩阵（关键步骤） ======
    U_noAI_2D = reshape(U_noAI, [], 6);   % 2091 × 6
    U_AI_2D   = reshape(U_AI,   [], 6);   % 2091 × 6
    AI_2D     = reshape(AInum,  [], 1);   % 2091 × 1

    %% ====== 输出：ΔU = U_AI − U_noAI ======
    U_diff_table = nan(6,6);

    disp('按 AI 最优等级分组求 ΔU_k ...');

    for k = 0:5
        mask = (AI_2D == k);   % 2091×1 logical vector

        U0 = U_noAI_2D(mask,:);   % N_k × 6
        U1 = U_AI_2D(mask,:);     % N_k × 6

        dU = U1 - U0;

        U_diff_table(k+1,:) = mean(dU,1,'omitnan');
    end

    %% ====== 输出结果 ======
    disp('====== 六个等级的效用变化（有AI − 无AI）======');
    disp(array2table(U_diff_table, ...
        'VariableNames', strcat("dU", string(0:5)) ));

end

%% =============================================================
%% 将 AI bestLevel 从 cell 转为 double
%% =============================================================
function AInum = convert_AI(AImat)
    [n,m] = size(AImat);
    AInum = nan(n,m);
    for i = 1:n
        for j = 1:m
            v = AImat{i,j};
            if ischar(v)
                if strcmp(v,'NaN')
                    AInum(i,j)=nan;
                else
                    AInum(i,j)=str2double(v);
                end
            else
                AInum(i,j)=v;
            end
        end
    end
end
