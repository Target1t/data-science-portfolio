%% Universality Check: Validating Strategic Rationale across the Parameter Space
clear; clc;

% 1. Load Data
X_all = readmatrix('X_fixedR03_allLevels.xlsx');    % 2091 rows, 36 columns
AI_grid = readmatrix('AI_bestLevel_grid.xlsx');     % 52x42 (with labels)

alphaVals = linspace(-1, 1, 41);
dcVals    = linspace(0, 5, 51);

% Results counters
results = struct('AI1', struct('count',0, 'pass',0), ...
                 'AI5', struct('count',0, 'pass',0), ...
                 'AI3', struct('count',0, 'pass',0), ...
                 'AI4', struct('count',0, 'pass',0));

fprintf('=== Starting Universality Check ===\n');

for j = 1:length(alphaVals)
    for i = 1:length(dcVals)
        currRow = (j-1) * 51 + i;
        AI_opt = AI_grid(i+1, j+1);
        
        % Extract distributions (6 levels each)
        get_dist = @(ai) X_all(currRow, (ai*6+1):(ai*6+6));
        
        % Check Hypotheses based on Optimal Level
        switch AI_opt
            case 1 % Hypothesis 1: AI=1 vs AI=5 for High-Tier (4/5)
                x1 = get_dist(1); x5 = get_dist(5);
                results.AI1.count = results.AI1.count + 1;
                if (x1(5)+x1(6)) > (x5(5)+x5(6)), results.AI1.pass = results.AI1.pass + 1; end
                
            case 5 % Hypothesis 2: AI=5 vs AI=1 for Mid-Tier (2/3)
                x5 = get_dist(5); x1 = get_dist(1);
                results.AI5.count = results.AI5.count + 1;
                if (x5(3)+x5(4)) > (x1(3)+x1(4)), results.AI5.pass = results.AI5.pass + 1; end
                
            case 3 % Hypothesis 3: AI=3 vs AI=4 for Mid-Tier (2/3)
                x3 = get_dist(3); x4 = get_dist(4);
                results.AI3.count = results.AI3.count + 1;
                if (x3(3)+x3(4)) > (x4(3)+x4(4)), results.AI3.pass = results.AI3.pass + 1; end
                
            case 4 % Hypothesis 4: AI=4 vs AI=3 for High-Tier (4/5)
                x4 = get_dist(4); x3 = get_dist(3);
                results.AI4.count = results.AI4.count + 1;
                if (x4(5)+x4(6)) > (x3(5)+x3(6)), results.AI4.pass = results.AI4.pass + 1; end
        end
    end
end

% Print Summary
fprintf('\n--- Universality Check Summary ---\n');
f_print = @(tag, res) fprintf('%s: Pass Rate %.1f%% (%d/%d)\n', tag, (res.pass/res.count)*100, res.pass, res.count);
f_print('AI=1 (High-Tier Advantage vs AI=5)', results.AI1);
f_print('AI=5 (Mid-Tier Support vs AI=1)  ', results.AI5);
f_print('AI=3 (Mid-Tier Bridging vs AI=4) ', results.AI3);
f_print('AI=4 (High-Tier Anchor vs AI=3)  ', results.AI4);