clear; clc;
% Folder containing the CSV files
folderPath = "./";  % or set as string, e.g., 'C:\path\to\your\folder'

% List all matching CSV files
fileList = dir(fullfile(folderPath, '*_predictions.csv'));

for k = 1:length(fileList)
    % Extract file name
    fileName = fileList(k).name;
    
    % Extract variable name (middle part)
    parts = split(fileName, '_');
    varName = [parts{2}, '_', extractBefore(parts{3}, 'C')];  % US06_n10

    % Read file into table
    filePath = fullfile(folderPath, fileName);
    T = readtable(filePath, 'VariableNamingRule','preserve');

    % Assign to workspace with dynamic name
    assignin('base', varName, T);
end

%%
%% Load Error Analysis and Plot

% Define endpoint lookup table
temps = [-20, -10, 0, 10, 25, 40];
cycles = {'HWFET', 'LA92', 'UDDS', 'US06'};
endpoints = [
    24677, 44378, 68848, 21982;
    27258, 48949, 77143, 22396;
    19770, 54806, 46000, 19585;
    26995, 55484, 82120, 22581;
    26728, 53065, 83226, 21532;
    27995, 55115, 84194, 23537
];

% Output folder for plots
outDir = 'plots_E66_FNN';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Error summary table init
error_summary = {};

% Iterate over all temperature and drive cycle combos
for t = 1:numel(temps)
    for c = 1:numel(cycles)
        temp = temps(t);
        cycle = cycles{c};

        % Construct variable name
        % Construct variable name
        if temp < 0
            temp_str = ['n' num2str(abs(temp))];
        else
            temp_str = num2str(temp);
        end
        varname = [cycle '_' temp_str];


        % Check if variable exists
        if ~evalin('base', sprintf('exist(''%s'', ''var'')', varname))
            warning('Skipping missing variable: %s', varname);
            continue;
        end

        % Load variable and extract values
        data = evalin('base', varname);
        N = min(endpoints(t, c), height(data));  % Safe clipping
        trueCol = localE66Column(data, {'True_Voltage', 'True Voltage (V)', 'TrueVoltage'});
        predCol = localE66Column(data, {'Predicted_Voltage', 'Predicted Voltage (V)', 'PredictedVoltage'});
        errCol  = localE66Column(data, {'Error (mV)', 'Error_mV', 'Error mV'});

        y_true = trueCol(1:N);
        y_pred = predCol(1:N);
        errors = errCol(1:N);

        % Error metrics
        rmse = sqrt(mean(errors.^2));
        maxe = max(abs(errors));
        error_summary = [error_summary; {cycle, temp, rmse, maxe}];

        % Plotting
        % ---- Create figure for publication ----
        h = figure('Units', 'inches', 'Position', [1, 1, 7.5, 6], 'Color', 'w');
        x_hr = (0:N-1) / 3600;
        
        % ---- Top subplot – Measured vs. Modeled ----
        subplot(2,1,1);
        plot(x_hr, y_true, 'k-', 'LineWidth', 1.5); hold on;
        plot(x_hr, y_pred, 'r--', 'LineWidth', 1.5);
        legend({'Measured', 'Modelled'}, 'FontSize', 12, 'Location', 'best');
        ylabel('Voltage (V)', 'FontSize', 14, 'FontWeight', 'bold');
        title(sprintf('%s at %d°C', cycle, temp), 'FontSize', 14, 'FontWeight', 'bold');
        set(gca, 'FontSize', 12, 'LineWidth', 1);
        
        % ---- Bottom subplot – Error ----
        subplot(2,1,2);
        plot(x_hr, errors, 'b', 'LineWidth', 1.5); hold on;
        yline(100, '--k', '100 mV', 'LabelVerticalAlignment','bottom', 'FontSize',10);
        yline(-100, '--k', '-100 mV', 'LabelVerticalAlignment','top', 'FontSize',10);
        yline(200, ':r', '200 mV', 'LabelVerticalAlignment','bottom', 'FontSize',10);
        yline(-200, ':r', '-200 mV', 'LabelVerticalAlignment','top', 'FontSize',10);
        xlabel('Time (h)', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel('Error (mV)', 'FontSize', 14, 'FontWeight', 'bold');
        ylim([-500 500]);
        title(sprintf('RMSE | %.0f mV, Max Error | %.0f mV', rmse, maxe), ...
              'FontSize', 14, 'FontWeight', 'bold');
        set(gca, 'FontSize', 12, 'LineWidth', 1);
        
        % ---- Tight layout (manual spacing if needed) ----
        set(gcf, 'PaperPositionMode', 'auto');
        
        % ---- Save plots in multiple formats ----
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end

        savefig(h, fullfile(outDir, [varname '.fig']));
        %saveas(h, fullfile(outDir, [varname '.png']));
        %print(h, fullfile(outDir, [varname '.pdf']), '-dpdf', '-r300');  % High-res vector
        % print(h, fullfile(outDir, [varname '.eps']), '-depsc', '-r300');  % EPS if needed
        
        close(h);  % Uncomment to suppress GUI display after saving

    end
end

% Convert summary to table and export
E66_Voltage_estimation_error_table = cell2table(error_summary, ...
    'VariableNames', {'DriveCycle', 'Temp', 'RMSE_mV', 'MaxError_mV'});
assignin('base', 'E66_Voltage_estimation_error_table_FNN', E66_Voltage_estimation_error_table);
%
writetable(E66_Voltage_estimation_error_table, 'E66_Voltage_estimation_error_table_FNN.csv');

%%
% Assumes 'E66_Voltage_estimation_error_table_FNN' is in workspace

% Extract unique temperatures and cycles
temps = unique(E66_Voltage_estimation_error_table.Temp);
cycles = {'HWFET', 'LA92', 'UDDS', 'US06'};
temps_str = arrayfun(@(t) sprintf('%d\\circC', t), temps, 'UniformOutput', false);

% Initialize matrices
rmse_mat = zeros(numel(temps), numel(cycles));
maxe_mat = zeros(numel(temps), numel(cycles));

% Fill matrices
for k = 1:height(E66_Voltage_estimation_error_table)
    row = E66_Voltage_estimation_error_table(k, :);
    r = find(temps == row.Temp);
    c = find(strcmp(cycles, row.DriveCycle));
    rmse_mat(r, c) = row.RMSE_mV;
    maxe_mat(r, c) = row.MaxError_mV;
end

%
% === Setup Plot ===
figure('Units', 'inches', 'Position', [1 1 7.5 5], 'Color', 'w');
hold on;

groupWidth = 0.75;
barWidth = groupWidth / numel(cycles);
colors = lines(numel(cycles));
labelYOffset = 15;
cycleLabelYOffset = 15;  % moved up from -40

rmseColor = [0.6, 0, 0.1];       % Deep red/maroon (RMSE)
maxeColor = [0.2, 0.5, 0.9];     % Rich sky blue (Max Error)

% === Plot Bars and Annotate ===
for c = 1:numel(cycles)
    x = (1:numel(temps)) + (c - (numel(cycles)+1)/2)*barWidth;

    % RMSE with border
    bar(x, rmse_mat(:,c), barWidth, 'FaceColor', rmseColor, ...
        'EdgeColor', 'k', 'LineWidth', 0.5);
    
    % Max Error with transparency and border
    bar(x, maxe_mat(:,c), barWidth, 'FaceColor', maxeColor, ...
        'EdgeColor', 'k', 'LineWidth', 0.5, 'FaceAlpha', 0.35);


    for i = 1:numel(x)
        text(x(i), rmse_mat(i,c) + labelYOffset, sprintf('%.0f', rmse_mat(i,c)), ...
            'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', 'k');
        text(x(i), maxe_mat(i,c) + labelYOffset, sprintf('%.0f', maxe_mat(i,c)), ...
            'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.1 0.1 0.6]);
        % Drive cycle label (near bottom of bars)
        text(x(i), -20, cycles{c}, ...  % was ~15 earlier
     'FontSize', 9, 'HorizontalAlignment', 'right', ...
     'Rotation', 45, 'Color', [0.3 0.3 0.3]);
    end
end

% === Add temperature labels and vertical lines ===
yTop = ceil(max(maxe_mat(:)) + 100);
for t = 1:numel(temps)
    % Dashed separator line
    if t > 1
        xline(t - 0.5, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
    end
    % Center x of group
    cycleOffsets = ((1:numel(cycles)) - (numel(cycles)+1)/2) * barWidth;
    centerX = mean(t + cycleOffsets);
    
    % Temperature label near top
    text(centerX, yTop * 0.95, temps_str{t}, ...
         'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% === Axes and Style ===
xlim([0.5, numel(temps) + 0.5]);
ylim([0, yTop]);
set(gca, 'XTick', []);  % Fully remove x-tick labels and their space
ylabel('Error (mV)', 'FontSize', 13, 'FontWeight', 'bold');
title('Voltage Estimation Error by Drive Cycle & Temperature', ...
      'FontSize', 14, 'FontWeight', 'bold');
legend({'RMSE', 'Max Error'}, 'FontSize', 11, 'Location', 'northeast');

set(gca, 'FontSize', 11, 'LineWidth', 1);
grid on;
% Optional Export
savefig(fullfile(outDir, 'voltage_error_barplot_FNN.fig'));

%% -----------------------------------------------
% P95 / P99 |error| (mV) heatmaps with value labels
% -----------------------------------------------
temps_u  = temps;                  % reuse your temps/cycles
cycles_u = cycles;

p95_mat = nan(numel(temps_u), numel(cycles_u));
p99_mat = nan(numel(temps_u), numel(cycles_u));

for ti = 1:numel(temps_u)
    for ci = 1:numel(cycles_u)
        temp  = temps_u(ti);
        cycle = cycles_u{ci};
        % varname in your script: e.g., 'US06_n20' or 'UDDS_25'
        if temp < 0, temp_str = ['n' num2str(abs(temp))]; else, temp_str = num2str(temp); end
        varname = [cycle '_' temp_str];

        if ~evalin('base', sprintf('exist(''%s'',''var'')', varname)), continue; end
        data = evalin('base', varname);
        N = min(endpoints(ti, ci), height(data));
        if N < 1, continue; end

        errCol = localE66Column(data, {'Error (mV)', 'Error_mV', 'Error mV'});
        err_abs = abs(errCol(1:N));
        p95_mat(ti, ci) = prctile(err_abs, 95);
        p99_mat(ti, ci) = prctile(err_abs, 99);
    end
end

% ---- Plot nicely with shared color scale and adaptive labels, saved separately
xTicks = 1:numel(temps_u);
yTicks = 1:numel(cycles_u);
xLabs  = compose('%d^\\circC', temps_u);

allVals = [p95_mat(:); p99_mat(:)];
cl = [min(allVals,[],'omitnan') max(allVals,[],'omitnan')];
cl = [10*floor(cl(1)/10), 10*ceil(cl(2)/10)];  % round to 10 mV

if ~exist(outDir,'dir'), mkdir(outDir); end

fig95 = figure('Units','inches','Position',[1 1 5.0 4.6],'Color','w');
imagesc(p95_mat'); axis tight;
ax = gca;
set(ax,'XTick',xTicks,'XTickLabel',xLabs, ...
       'YTick',yTicks,'YTickLabel',cycles_u, ...
       'TickDir','out','Box','on','Layer','top', ...
       'FontWeight','bold','XColor','k','YColor','k');
colormap(turbo); caxis(cl);
grid(ax,'off');
cb = colorbar; cb.Label.String = 'mV'; cb.TickDirection = 'out';
cb.Color = 'k'; cb.FontWeight = 'bold';
cb.Label.Color = 'k'; cb.Label.FontWeight = 'bold';
title('P95 |error| (mV)','FontWeight','bold','Color','k');
localLabelHeatmapCells(p95_mat, '%.0f');
savefig(fig95, fullfile(outDir,'voltage_error_p95_heatmap_FNN.fig'));
saveas(fig95, fullfile(outDir,'voltage_error_p95_heatmap_FNN.png'));

fig99 = figure('Units','inches','Position',[1 1 5.0 4.6],'Color','w');
imagesc(p99_mat'); axis tight;
ax = gca;
set(ax,'XTick',xTicks,'XTickLabel',xLabs, ...
       'YTick',yTicks,'YTickLabel',cycles_u, ...
       'TickDir','out','Box','on','Layer','top', ...
       'FontWeight','bold','XColor','k','YColor','k');
colormap(turbo); caxis([0,150]);
grid(ax,'off');
cb = colorbar; cb.Label.String = 'mV'; cb.TickDirection = 'out';
cb.Color = 'k'; cb.FontWeight = 'bold';
cb.Label.Color = 'k'; cb.Label.FontWeight = 'bold';
title('P99 |error| (mV)','FontWeight','bold','Color','k');
localLabelHeatmapCells(p99_mat, '%.0f');
savefig(fig99, fullfile(outDir,'voltage_error_p99_heatmap_FNN.fig'));
saveas(fig99, fullfile(outDir,'voltage_error_p99_heatmap_FNN.png'));


%% -----------------------------------------------
% Average RMSE by Temperature
% -----------------------------------------------
% This section assumes that the 'E66_Voltage_estimation_error_table_FNN'
% from the previous parts of the script is available in the workspace.

% Check if the error table exists before proceeding
if ~exist('E66_Voltage_estimation_error_table_FNN', 'var')
    warning('The table E66_Voltage_estimation_error_table does not exist. Please run the previous sections of the script first.');
    return;
end

function values = localE66Column(T, candidates)
    names = string(T.Properties.VariableNames);
    normalizedNames = lower(regexprep(names, '[^a-zA-Z0-9]', ''));

    if ischar(candidates) || isstring(candidates)
        candidates = cellstr(candidates);
    end

    for k = 1:numel(candidates)
        target = lower(regexprep(string(candidates{k}), '[^a-zA-Z0-9]', ''));
        idx = find(strcmpi(normalizedNames, target), 1);
        if ~isempty(idx)
            values = T.(names(idx));
            return;
        end
    end

    error('E66 column not found. Tried: %s', strjoin(string(candidates), ', '));
end

function localLabelHeatmapCells(M, fmt)
    [nRows, nCols] = size(M');
    clim = get(gca, 'CLim');
    if ~all(isfinite(clim))
        lo = min(M(:), [], 'omitnan');
        hi = max(M(:), [], 'omitnan');
        if ~isfinite(lo) || ~isfinite(hi)
            clim = [0 1];
        else
            clim = [lo hi];
        end
    end

    hold on;
    for r = 1:nRows
        for c = 1:nCols
            val = M(c, r);
            if isnan(val)
                continue;
            end
            tNorm = (val - clim(1)) / max(eps, clim(2) - clim(1));
            txtColor = [0.05 0.05 0.05];
            if tNorm <= 0.55
                txtColor = [0.98 0.98 0.98];
            end
            text(c, r, sprintf(fmt, val), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontWeight','bold','FontSize',9,'Color',txtColor);
        end
    end

    set(gca,'YDir','normal');
    for xx = 0.5:1:nCols+0.5
        line([xx xx], [0.5 nRows+0.5], 'Color', [.9 .9 .9]);
    end
    for yy = 0.5:1:nRows+0.5
        line([0.5 nCols+0.5], [yy yy], 'Color', [.9 .9 .9]);
    end
end

% Use groupsummary instead of groupfilter to calculate the mean of RMSE_mV
% for each temperature group. This is the correct function for this task.
avg_rmse_by_temp = groupsummary(E66_Voltage_estimation_error_table, 'Temp', 'mean', 'RMSE_mV');

% Create a new figure for this plot
figure('Units', 'inches', 'Position', [1 1 6 4], 'Color', 'w');

% Plot the average RMSE values as a bar chart
bar(avg_rmse_by_temp.Temp, avg_rmse_by_temp.mean_RMSE_mV, 'FaceColor', [0 0.447 0.741], 'EdgeColor', 'k');
hold on;
grid on;

% Add the trend line on top of the bars
plot(avg_rmse_by_temp.Temp, avg_rmse_by_temp.mean_RMSE_mV, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'MarkerSize', 6);

% Add the numerical values on top of each bar
for i = 1:height(avg_rmse_by_temp)
    text(avg_rmse_by_temp.Temp(i), avg_rmse_by_temp.mean_RMSE_mV(i) + 1, ...
         sprintf('%.1f', avg_rmse_by_temp.mean_RMSE_mV(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9, 'FontWeight', 'bold');
end

% Add labels and a title to make the plot clear and informative.
% The x-axis label has been updated to use standard characters.
xlabel('Temperature (°C)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Average RMSE (mV)', 'FontSize', 12, 'FontWeight', 'bold');
title('Average RMSE by Temperature', 'FontSize', 14, 'FontWeight', 'bold');
% Set the y-axis limit to make the plot look better
ylim([0 40]);

% Improve the visual appearance of the axes
set(gca, 'FontSize', 10, 'LineWidth', 1);

% Ensure the output directory exists before saving
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Save the plot in both .fig and .png formats
savefig(gcf, fullfile(outDir, 'average_rmse_by_temp_FNN.fig'));
saveas(gcf, fullfile(outDir, 'average_rmse_by_temp.png'));

%%
%% ==== Present vs Reference RMSE (overlayed grouped bars with %Δ labels) ====

% Load reference table (baseline) and align to existing temps/cycles layout
Tref = readtable('E^^_Voltage_estimation_error_table_3e-3.csv', ...
                 'VariableNamingRule','preserve');

% Build reference RMSE matrix aligned with temps_u & cycles_u (already in workspace)
rmse_ref_mat = nan(numel(temps_u), numel(cycles_u));
for k = 1:height(Tref)
    r = find(temps_u == Tref.Temp(k));
    c = find(strcmp(cycles_u, Tref.DriveCycle{k}));
    if ~isempty(r) && ~isempty(c)
        rmse_ref_mat(r,c) = Tref.RMSE_mV(k);
    end
end

% Figure: grouped by temperature; within each group bars per cycle
figure('Units','inches','Position',[1 1 7.5 4.8],'Color','w'); hold on
gw = 0.75;                      % group width
bw = gw/numel(cycles_u);        % bar width per cycle within a group

% Colors/styles (print-friendly)
col_present = [0.70 0.70 0.70]; % solid light gray
col_ref     = [0.30 0.30 0.30]; % darker gray (semi-transparent overlay)

% Draw present (solid) and overlay reference (shaded) at the SAME x locations
for c = 1:numel(cycles_u)
    x = (1:numel(temps_u)) + (c-(numel(cycles_u)+1)/2)*bw;

    % Present RMSE (solid, opaque — blue)
    bar(x, rmse_mat(:,c), 0.95*bw, ...
        'FaceColor', [0.16 0.44 0.84], 'FaceAlpha', 0.95, ...
        'EdgeColor', [0.10 0.28 0.54], 'LineWidth', 0.6);
    
    % Reference RMSE (overlay, semi-transparent — orange)
    bar(x, rmse_ref_mat(:,c), 0.65*bw, ...
        'FaceColor', [0.90 0.47 0.13], 'FaceAlpha', 0.55, ...
        'EdgeColor', [0.60 0.32 0.09], 'LineWidth', 0.7);

end

% X-ticks: cycle names under each bar (rotated), as in your prior grouped plot
off = ((1:numel(cycles_u)) - (numel(cycles_u)+1)/2) * bw;
xt = []; labs = {};
for t = 1:numel(temps_u)
    xt   = [xt, t + off]; %#ok<AGROW>
    labs = [labs, cycles_u]; %#ok<AGROW>
end
set(gca,'XTick',xt,'XTickLabel',labs,'XTickLabelRotation',45, ...
        'TickDir','out','Box','on','Layer','top');
grid on

% Y-axis limits & headroom
maxVal  = max([rmse_mat(:); rmse_ref_mat(:)], [], 'omitnan');
yPadTop = max(30, 0.12*maxVal);
ylim([0, ceil(maxVal + yPadTop)]); xlim([0.5, numel(temps_u)+0.5]);

% Vertical separators between temperature groups + temperature labels on top
for t = 1:numel(temps_u)
    if t>1, xline(t-0.5, ':', 'Color',[0.85 0.85 0.85]); end
    cx = mean(t + off);
    text(cx, ylim(gca)*[0;1] - 0.35*yPadTop, sprintf('%d^{\\circ}C', temps_u(t)), ...
        'horiz','center','FontWeight','bold','Color',[0.25 0.25 0.25]);
end

ylabel('RMSE (mV)');
title('Voltage RMSE — Present vs Reference (overlay)');

% --- %Δ labels (present vs reference) over each (temp,cycle) ---
yl = ylim; yspan = yl(2)-yl(1); ypad = 0.035*yspan;
for t = 1:numel(temps_u)
    for c = 1:numel(cycles_u)
        cur = rmse_mat(t,c);
        ref = rmse_ref_mat(t,c);
        if ~isnan(cur) && ~isnan(ref) && ref > 0
            dperc = 100*(cur - ref)/ref;     % as requested: (present - ref) in %
            xpos  = t + (c-(numel(cycles_u)+1)/2)*bw;
            ypos  = max(cur, ref) + ypad;
            col   = [0.10 0.55 0.10];        % green if improved (negative %)
            if dperc >= 0, col = [0.75 0.10 0.10]; end  % red if worse (positive %)
            text(xpos, ypos, sprintf('%+.0f%%', dperc), ...
                'HorizontalAlignment','center','VerticalAlignment','bottom', ...
                'FontWeight','bold','Color', col, 'FontSize', 8);
        end
    end
end

legend({'fc1=0.05, fc2=0.005, fc3=0.0005','fc=0.005'}, 'Location','northeast', 'Box','off');

