function whenNoise(d, xvar, xlabel_str, out_path, legend_position, smooth, tol, ylim_add)
arguments
    d
    xvar
    xlabel_str = 'x'
    out_path = ''
    legend_position = 'northeast'
    smooth (1,1) logical = false
    tol=1e-02
    ylim_add=0.05
end
% WHENNOISE
% Region where lambda*(xvar, Delta) > tol from simulation results d.
% - x-axis: xvar (field/column in d)
% - y-axis: Delta
% - Value: indicator {noise > tol}, where 'noise' in d is optimal lambda*
%   and tol defaults to 1e-2 (or d(1).DIGITS if present).
%
% If smooth==true:
%   - Identify columns (in x) where ZQ>tol is nonempty and compute column-wise
%     lower/upper edges via min/max over y (per column).
%   - Fit quadratic y(x)=a x^2 + b x + c to each edge (constant curvature).
%   - Fill region between the two smoothed edges and draw only those edges.

%% ---------- Extract data ----------
[X, Y, L] = extractXYL(d, xvar);   % X: xvar; Y: Delta; L: lambda*

if isempty(X) || isempty(Y) || isempty(L)
    error('Could not extract xvar, Delta, and noise from d.');
end

% Default tolerance 1e-2, unless DIGITS is present and nonempty
if isstruct(d) && ~isempty(d)
    if isfield(d, 'DIGITS') && ~isempty(d(1).DIGITS)
        tol = double(d(1).DIGITS);
    end
elseif istable(d) && any(strcmpi(d.Properties.VariableNames,'DIGITS'))
    tmp = d.DIGITS;
    if ~isempty(tmp) && isnumeric(tmp)
        tol = double(tmp(1));
    end
end
tol = max(1e-12, tol);

% Aggregate duplicates by mean(L)
[U, ~, idx] = unique([X(:), Y(:)], 'rows', 'stable');
Lmean = accumarray(idx, L(:), [], @mean);
X = U(:,1); Y = U(:,2); L = Lmean;

%% ---------- Interpolate onto grid ----------
Nx = min(400, max(100, 3*numel(unique(X))));
Ny = min(400, max(100, 3*numel(unique(Y))));
xq = linspace(min(X), max(X), Nx);
yq = linspace(min(Y), max(Y), Ny);
[XQ, YQ] = meshgrid(xq, yq);

F  = scatteredInterpolant(X, Y, L, 'linear', 'nearest');
ZQ = F(XQ, YQ);                     % lambda* on grid
M  = ZQ > tol;                      % mask of positive noise (tol)

%% ---------- Figure & axes (sizes in centimeters, LaTeX text) ----------
fwidth = 8;
fheight=6;
fig = figure('Color','w', 'Units','centimeters', 'Position',[2 2 fwidth fheight]);
ax  = axes('Parent',fig, 'Units','centimeters');

% Fix the plotting area (margins in cm)
left = fwidth * 0.1; bottom = fheight * 0.1;
ax.Position = [left, bottom, fwidth*0.88, fheight * 0.88];

hold(ax,'on');
set(ax, 'Box','on', 'Layer','top', 'TickLabelInterpreter','latex');
ax.FontSize = 6;  % tick labels

xlabel(ax, xlabel_str, 'Interpreter','latex', 'FontSize', 8);
ylabel(ax, '$\Delta$',   'Interpreter','latex', 'FontSize', 8);

%% ---------- Draw region & edges ----------
if ~smooth
    % --- Raw filled region via mask and raw boundary at level tol
    Cmap = [1 1 1; 0.9 0.9 0.9]; % white & light gray fill for M==1
    contourf(ax, XQ, YQ, double(M), [0.5 1.5], 'LineStyle','none');
    colormap(fig, Cmap);

    % Raw boundary (lambda* = tol)
    contour(ax, XQ, YQ, ZQ, [tol tol], 'k-', 'LineWidth', 1.2);

    % Legend
    hRegion = patch(ax, NaN, NaN, [0.9 0.9 0.9], 'EdgeColor','none', ...
        'DisplayName', '$\lambda^*>0$');
    hU = plot(ax, NaN, NaN, 'k-', 'LineWidth', 1.2, 'DisplayName', '$\bar{\Delta}$');
    hL = plot(ax, NaN, NaN, 'k-', 'LineWidth', 1.2, 'DisplayName', '$\underline{\Delta}$');
    lg = legend(ax, [hRegion hU hL], 'Location', legend_position, ...
        'Interpreter','latex', 'FontSize', 8);
else
    % --- Smoothed edges & filled region between them (no raw edges/contours)
    % For each x-column, find y-lower/y-upper inside region
    hasAny = any(M, 1);             % 1 x Nx
    runs   = contiguousRuns(find(hasAny));
    gray   = [0.9 0.9 0.9];

    hRegionList = gobjects(0);
    hUpperList  = gobjects(0);
    hLowerList  = gobjects(0);

    for r = 1:size(runs,1)
        j1 = runs(r,1); j2 = runs(r,2);
        jj = j1:j2;

        % Column-wise lower/upper edges from mask
        ylow  = nan(1, numel(jj));
        yhigh = nan(1, numel(jj));
        for k = 1:numel(jj)
            col = jj(k);
            rows = find(M(:,col));
            if ~isempty(rows)
                ylow(k)  = yq(min(rows));
                yhigh(k) = yq(max(rows));
            end
        end

        % Remove NaNs (just in case)
        msk = ~(isnan(ylow) | isnan(yhigh));
        if nnz(msk) < 3
            continue; % not enough points to fit quadratics
        end

        xcol  = xq(jj(msk));
        ylow  = ylow(msk);
        yhigh = yhigh(msk);

        % Quadratic fits (constant curvature per edge)
        pL = polyfit(xcol, ylow,  2);
        pU = polyfit(xcol, yhigh, 2);

        % Evaluate smoothed edges on a denser x-grid over this run
        xs = linspace(min(xcol), max(xcol), max(100, 3*numel(xcol)));
        yL = polyval(pL, xs);
        yU = polyval(pU, xs);

        % Ensure lower <= upper (robustness)
        swapIdx = yL > yU;
        if any(swapIdx)
            tmp = yL(swapIdx); yL(swapIdx) = yU(swapIdx); yU(swapIdx) = tmp;
        end

        % Fill region between smoothed edges
        hRegionList(end+1) = patch(ax, [xs, fliplr(xs)], [yL, fliplr(yU)], gray, ...
            'EdgeColor','none', 'FaceAlpha', 1); %#ok<AGROW>

        % Draw smoothed upper & lower edges (no raw edges)
        hUpperList(end+1) = plot(ax, xs, yU, 'k-', 'LineWidth', 1.2, 'DisplayName', '$\bar{\Delta}$'); %#ok<AGROW>
        hLowerList(end+1) = plot(ax, xs, yL, 'k--', 'LineWidth', 1.2, 'DisplayName', '$\underline{\Delta}$'); %#ok<AGROW>
    end

    % Legend (one representative handle of each type)
    if ~isempty(hRegionList)
        hRegion = hRegionList(1);
        hU = hUpperList(min(1,numel(hUpperList)));
        hL = hLowerList(min(1,numel(hLowerList)));
        set(hRegion, 'DisplayName', '$\lambda^*>0$');
        set(hU,      'DisplayName', '$\bar{\Delta}$');
        set(hL,      'DisplayName', '$\underline{\Delta}$');
        lg = legend(ax, [hRegion hU hL], 'Location', legend_position, ...
            'Interpreter','latex', 'FontSize', 8);
    end
end

axis(ax,'tight');
set(ax,'YDir','normal');
grid(ax,'on');
ylim([min(d.Delta)-0.05, max(d.Delta)+ylim_add])

% Optional export
if ~isempty(out_path)
    [fp,fn,fe] = fileparts(out_path);
    if isempty(fe), fe = '.png'; out_path = fullfile(fp, [fn fe]); end
    if ~isempty(fp) && ~exist(fp,'dir'); mkdir(fp); end
    try
        exportgraphics(fig, out_path, 'Resolution', 300);
    catch
        warning('Failed to export figure to %s', out_path);
    end
end
end

%% =============== Helpers ===============

function runs = contiguousRuns(idx)
% Given a vector of integer indices, return [start end] rows for contiguous runs.
if isempty(idx), runs = zeros(0,2); return; end
d = diff(idx(:).');
brk = [true, d>1, true];
loc = find(brk);
runs = [idx(loc(1:end-1)); idx(loc(2:end)-1)].';
end

function [X, Y, L] = extractXYL(d, xvar)
% Extract X = d.(xvar), Y = d.Delta, L = d.noise from struct array or table.
% Returns column vectors (double). Missing/NaN rows are removed.

X = []; Y = []; L = [];
if istable(d)
    vn = d.Properties.VariableNames;
    if ~any(strcmp(vn, xvar)), error('xvar "%s" not found in table.', xvar); end
    if ~any(strcmpi(vn, 'Delta')), error('Field "Delta" not found in table.'); end
    if any(strcmpi(vn, 'noise'))
        lcol = 'noise';
    elseif any(strcmpi(vn, 'lambda')) || any(strcmpi(vn, 'lambdaStar'))
        lcol = vn{find(strcmpi(vn,'lambda') | strcmpi(vn,'lambdaStar'), 1, 'first')};
    else
        error('Could not find "noise" (lambda*) column in table.');
    end
    X = double(d.(xvar));   Y = double(d.('Delta'));   L = double(d.(lcol));
elseif isstruct(d)
    if ~isfield(d, xvar), error('xvar "%s" not found in struct.', xvar); end
    if ~isfield(d, 'Delta'), error('Field "Delta" not found in struct.'); end
    if isfield(d, 'noise')
        lfield = 'noise';
    elseif isfield(d, 'lambda')
        lfield = 'lambda';
    elseif isfield(d, 'lambdaStar')
        lfield = 'lambdaStar';
    else
        error('Could not find "noise" (lambda*) field in struct.');
    end
    X = double([d.(xvar)]);   Y = double([d.Delta]);   L = double([d.(lfield)]);
else
    error('d must be a table or a struct array.');
end

X = X(:); Y = Y(:); L = L(:);
ok = ~(isnan(X) | isnan(Y) | isnan(L));
X = X(ok); Y = Y(ok); L = L(ok);
end
