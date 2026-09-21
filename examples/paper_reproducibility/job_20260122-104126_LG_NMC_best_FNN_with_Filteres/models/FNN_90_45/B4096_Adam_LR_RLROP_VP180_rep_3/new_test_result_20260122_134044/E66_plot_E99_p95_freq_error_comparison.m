clear; clc;
folderPath = "./";

fileList = dir(fullfile(folderPath, '*_predictions.csv'));

for k = 1:numel(fileList)
    fileName = fileList(k).name;
    parts = split(fileName, '_');

    % Find the temperature token (e.g., '0C', '10C', '40C', 'n10C')
    tempIdx = find(endsWith(parts, 'C'), 1, 'first');
    if isempty(tempIdx) || tempIdx == 1
        warning('Could not parse temp from file: %s', fileName);
        continue
    end

    drive = parts{tempIdx-1};                % e.g., 'US06', 'HWFET'
    tempStr = extractBefore(parts{tempIdx}, 'C');  % e.g., '0', '40', 'n10'

    varName = sprintf('%s_%s', drive, tempStr);    % e.g., 'US06_40'
    varName = matlab.lang.makeValidName(varName);  % safety

    T = readtable(fullfile(folderPath, fileName));
    assignin('base', varName, T);
end

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
outDir = 'matplots_e66_voltage_print_E66';
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
        y_true = data.True_Voltage(1:N);
        y_pred = data.Predicted_Voltage(1:N);
        errors = data.Error_mV_(1:N);

        % Error metrics
        rmse = sqrt(mean(errors.^2));
        maxe = max(abs(errors));
        error_summary = [error_summary; {cycle, temp, rmse, maxe}];

        % Plotting
        % ---- Create figure for publication ----
        h = figure('Units', 'inches', 'Position', [1, 1, 7.5, 6], 'Color', 'w');
        
        % ---- Top subplot – Measured vs. Modeled ----
        subplot(2,1,1);
        plot(1:N, y_true, 'k-', 'LineWidth', 1.5); hold on;
        plot(1:N, y_pred, 'r--', 'LineWidth', 1.5);
        legend({'Measured', 'Modelled'}, 'FontSize', 12, 'Location', 'best');
        ylabel('Voltage (V)', 'FontSize', 14, 'FontWeight', 'bold');
        title(sprintf('%s at %d°C', cycle, temp), 'FontSize', 14, 'FontWeight', 'bold');
        set(gca, 'FontSize', 12, 'LineWidth', 1);
        
        % ---- Bottom subplot – Error ----
        subplot(2,1,2);
        plot(1:N, errors, 'b', 'LineWidth', 1.5); hold on;
        yline(100, '--k', '100 mV', 'LabelVerticalAlignment','bottom', 'FontSize',10);
        yline(-100, '--k', '-100 mV', 'LabelVerticalAlignment','top', 'FontSize',10);
        yline(200, ':r', '200 mV', 'LabelVerticalAlignment','bottom', 'FontSize',10);
        yline(-200, ':r', '-200 mV', 'LabelVerticalAlignment','top', 'FontSize',10);
        xlabel('Sample', 'FontSize', 14, 'FontWeight', 'bold');
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
assignin('base', 'E66_Voltage_estimation_error_table', E66_Voltage_estimation_error_table);
%
writetable(E66_Voltage_estimation_error_table, 'E66_Voltage_estimation_error_table_FNN.csv');

%%
% Assumes 'E66_Voltage_estimation_error_table' is in workspace

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
savefig(fullfile(outDir, 'voltage_error_barplot.fig'));

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

        err_abs = abs(data.Error_mV_(1:N));
        p95_mat(ti, ci) = prctile(err_abs, 95);
        p99_mat(ti, ci) = prctile(err_abs, 99);
    end
end

% ---- Plot nicely with shared color scale and numeric labels
figure('Units','inches','Position',[1 1 8.8 4.6],'Color','w');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

xTicks = 1:numel(temps_u); yTicks = 1:numel(cycles_u);
xLabs  = compose('%d^\\circC', temps_u);

allVals = [p95_mat(:); p99_mat(:)];
cl = [min(allVals,[],'omitnan') max(allVals,[],'omitnan')];
cl = [10*floor(cl(1)/10), 10*ceil(cl(2)/10)];  % round to 10 mV

% --- P95 ---
nexttile;
imagesc(p95_mat'); axis tight; set(gca,'YDir','normal');
set(gca,'XTick',xTicks,'XTickLabel',xLabs, ...
        'YTick',yTicks,'YTickLabel',cycles_u, ...
        'TickDir','out','Box','on','Layer','top');
colormap(turbo); caxis(cl);
cb = colorbar; cb.Label.String = 'mV';
title('P95 |error| (mV)'); grid on;

% labels
for r = 1:numel(cycles_u)
    for c = 1:numel(temps_u)
        val = p95_mat(c, r); if isnan(val), continue; end
        text(c, r, sprintf('%.0f', val), 'horiz','center','vert','middle', ...
            'FontWeight','bold', 'Color', 'b', 'FontSize',9);
    end
end

% --- P99 ---
nexttile;
imagesc(p99_mat'); axis tight; set(gca,'YDir','normal');
set(gca,'XTick',xTicks,'XTickLabel',xLabs, ...
        'YTick',yTicks,'YTickLabel',cycles_u, ...
        'TickDir','out','Box','on','Layer','top');
colormap(turbo); caxis(cl);
cb = colorbar; cb.Label.String = 'mV';
title('P99 |error| (mV)'); grid on;

% labels
for r = 1:numel(cycles_u)
    for c = 1:numel(temps_u)
        val = p99_mat(c, r); if isnan(val), continue; end
        text(c, r, sprintf('%.0f', val), 'horiz','center','vert','middle', ...
            'FontWeight','bold', 'Color', 'b', 'FontSize',9);
    end
end

sgtitle('E66 Voltage |error| Quantiles');

if ~exist(outDir,'dir'), mkdir(outDir); end
savefig(fullfile(outDir,'voltage_error_p95_p99_heatmaps.fig'));
saveas(gcf, fullfile(outDir,'voltage_error_p95_p99_heatmaps.png'));


%% -----------------------------------------------
% Average RMSE by Temperature
% -----------------------------------------------
% This section assumes that the 'E66_Voltage_estimation_error_table'
% from the previous parts of the script is available in the workspace.

% Check if the error table exists before proceeding
if ~exist('E66_Voltage_estimation_error_table', 'var')
    warning('The table E66_Voltage_estimation_error_table does not exist. Please run the previous sections of the script first.');
    return;
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
savefig(gcf, fullfile(outDir, 'average_rmse_by_temp.fig'));

%% ==== Present vs Reference RMSE (overlayed grouped bars with %Δ labels) ====

% Load reference table (baseline) and align to existing temps/cycles layout
Tref = readtable('E66_Voltage_estimation_error_table_FNN', ...
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


%% ==== LSTM (present) vs FNN (reference) RMSE (overlayed grouped bars with %Δ labels) ====

% Load FNN reference table and align to existing temps/cycles layout
Tref = readtable('E66_Voltage_estimation_error_table_FNN', ...
                 'VariableNamingRule','preserve');

% Build FNN reference RMSE matrix aligned with temps_u & cycles_u (already in workspace)
rmse_ref_mat = nan(numel(temps_u), numel(cycles_u));
for k = 1:height(Tref)
    r = find(temps_u == Tref.Temp(k));
    c = find(strcmp(cycles_u, Tref.DriveCycle{k}));
    if ~isempty(r) && ~isempty(c)
        rmse_ref_mat(r,c) = Tref.RMSE_mV(k);   % FNN RMSE
    end
end

% rmse_mat is assumed to be ECM RMSE (present work) with same layout:
%   rows = temps_u, cols = cycles_u

% Figure: grouped by temperature; within each group bars per cycle
figure('Units','inches','Position',[1 1 7.5 4.8],'Color','w'); hold on
gw = 0.75;                      % group width
bw = gw/numel(cycles_u);        % bar width per cycle within a group

% Draw ECM (present) and overlay FNN (reference) at the SAME x locations
for c = 1:numel(cycles_u)
    x = (1:numel(temps_u)) + (c-(numel(cycles_u)+1)/2)*bw;

    % ECM RMSE (solid, opaque — blue)
    bar(x, rmse_mat(:,c), 0.95*bw, ...
        'FaceColor', [0.16 0.44 0.84], 'FaceAlpha', 0.95, ...
        'EdgeColor', [0.10 0.28 0.54], 'LineWidth', 0.6);

    % FNN RMSE (overlay, semi-transparent — orange)
    bar(x, rmse_ref_mat(:,c), 0.65*bw, ...
        'FaceColor', [0.90 0.47 0.13], 'FaceAlpha', 0.55, ...
        'EdgeColor', [0.60 0.32 0.09], 'LineWidth', 0.7);
end

% X-ticks: cycle names under each bar (rotated)
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
ylim([0, ceil(maxVal + yPadTop)]); 
xlim([0.5, numel(temps_u)+0.5]);

% Vertical separators between temperature groups + temperature labels on top
for t = 1:numel(temps_u)
    if t>1, xline(t-0.5, ':', 'Color',[0.85 0.85 0.85]); end
    cx = mean(t + off);
    text(cx, ylim(gca)*[0;1] - 0.35*yPadTop, sprintf('%d^{\\circ}C', temps_u(t)), ...
        'horiz','center','FontWeight','bold','Color',[0.25 0.25 0.25]);
end

ylabel('RMSE (mV)');
title('Voltage RMSE — LSTM (present) vs FNN (reference)');

% --- %Δ labels (ECM vs FNN) over each (temp,cycle) ---
% definition: dperc = 100 * (ECM - FNN) / FNN
%   >0  → ECM worse than FNN   (red)
%   <0  → ECM better than FNN  (green)
yl = ylim; yspan = yl(2)-yl(1); ypad = 0.035*yspan;
for t = 1:numel(temps_u)
    for c = 1:numel(cycles_u)
        cur = rmse_mat(t,c);        % ECM
        ref = rmse_ref_mat(t,c);    % FNN
        if ~isnan(cur) && ~isnan(ref) && ref > 0
            dperc = 100*(cur - ref)/ref;
            xpos  = t + (c-(numel(cycles_u)+1)/2)*bw;
            ypos  = max(cur, ref) + ypad;
            col   = [0.10 0.55 0.10];        % green if ECM improved (negative %)
            if dperc >= 0, col = [0.75 0.10 0.10]; end  % red if ECM worse (positive %)
            text(xpos, ypos, sprintf('%+.0f%%', dperc), ...
                'HorizontalAlignment','center','VerticalAlignment','bottom', ...
                'FontWeight','bold','Color', col, 'FontSize', 8);
        end
    end
end

legend({'LSTM (present work)','FNN (reference)'}, ...
       'Location','northeast', 'Box','off');
savefig(fullfile(outDir, 'LSTM_vs_FNN_error_bar_overlay.fig'));


%% ================= 2x2 compact RMSE line plots (journal-ready) =================

% ---- choose cycles & order ----
preferred = ["HWFET","LA92","UDDS","US06"];
cycles_plot = preferred(ismember(preferred, cycles_u));
if isempty(cycles_plot)
    cycles_plot = cycles_u;
end
idxC = arrayfun(@(s) find(cycles_u==s,1), cycles_plot);

% ---- shared Y limit across all panels ----
allVals = [rmse_LSTM(:,idxC); rmse_ECM(:,idxC); rmse_FNN(:,idxC)];
yMax = max(allVals(:), [], 'omitnan');
yLimShared = [0 ceil(1.10*yMax + 2)];

% ---- figure ----
fig = figure('Units','inches','Position',[1 1 6.6 3.6],'Color','w');

% ---- tiled layout (reserve TOP space for legend) ----
tl = tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');
tl.OuterPosition = [0.02 0.02 0.96 0.84];   % <-- IMPORTANT (legend space)

% ---- styles ----
cL = [0.20 0.45 0.70];   % LSTM
cE = [0.20 0.65 0.25];   % ECM
cF = [0.85 0.45 0.10];   % FNN

lw = 1.6;
ms = 4.6;

hL = gobjects(1); hE = gobjects(1); hF = gobjects(1);

% ---- plot panels ----
for p = 1:min(4,numel(idxC))
    c = idxC(p);
    ax = nexttile(tl,p); hold(ax,'on'); box(ax,'on');

    yL = rmse_LSTM(:,c);
    yE = rmse_ECM(:,c);
    yF = rmse_FNN(:,c);

    h1 = plot(ax, temps_u, yL,'-o','Color',cL,'LineWidth',lw,'MarkerSize',ms,...
        'MarkerFaceColor',cL,'MarkerEdgeColor',cL);
    h2 = plot(ax, temps_u, yE,'-^','Color',cE,'LineWidth',lw,'MarkerSize',ms,...
        'MarkerFaceColor',cE,'MarkerEdgeColor',cE);
    h3 = plot(ax, temps_u, yF,'-s','Color',cF,'LineWidth',lw,'MarkerSize',ms,...
        'MarkerFaceColor',cF,'MarkerEdgeColor',cF);

    if p==1, hL=h1; hE=h2; hF=h3; end

    title(ax, strrep(string(cycles_u(c)),"_","\_"), ...
        'FontWeight','bold','FontSize',9);

    ylim(ax,yLimShared);
    grid(ax,'on'); ax.GridAlpha = 0.10;

    set(ax,'FontName','Times New Roman','FontSize',9,...
        'TickDir','out','LineWidth',0.8,'Layer','top');

    % ---- reduce label clutter ----
    if ismember(p,[1 3])
        ylabel(ax,'RMSE (mV)');
    else
        ax.YTickLabel = [];
    end
    if ismember(p,[3 4])
        xlabel(ax,'Temperature (°C)');
    else
        ax.XTickLabel = [];
    end
end

% ---- legend placed in RESERVED TOP BAND ----
lg = legend([hL hE hF], ...
    {'LSTM','ECM','FNN'}, ...
    'Orientation','horizontal','Box','off');

lg.Units = 'normalized';
lg.Position = [0.18 0.90 0.64 0.07];   % inside reserved band
lg.FontSize = 9;

% ---- save (optional) ----
if exist('outDir','var') && isfolder(outDir)
    exportgraphics(fig,fullfile(outDir,'RMSE_LSTM_vs_FNN_vs_ECM_2x2.pdf'),...
        'ContentType','vector');
end

%% ==== Temperature-wise average RMSE: ECM vs FNN + % difference labels ====
% rmse_mat      : ECM RMSE (present), size = [numTemps x numCycles]
% rmse_ref_mat  : FNN RMSE (reference), same size
% temps_u       : vector of unique temperatures

% 1) Temperature-wise average RMSE for each method
ecm_avg_rmse = mean(rmse_mat,     2, 'omitnan');   % [numTemps x 1]
fnn_avg_rmse = mean(rmse_ref_mat, 2, 'omitnan');   % [numTemps x 1]

% 2) % difference (ECM vs FNN) at each temperature
%    dperc < 0 → ECM better (lower avg RMSE)
%    dperc > 0 → ECM worse (higher avg RMSE)
dperc_temp = 100 * (ecm_avg_rmse - fnn_avg_rmse) ./ fnn_avg_rmse;

% 3) Plot: grouped bars by temperature
figure('Units','inches','Position',[1 1 6.5 3.8],'Color','w'); hold on

% Grouped bar: one group per temperature, two bars (ECM, FNN)
B = bar(temps_u, [ecm_avg_rmse fnn_avg_rmse], 0.75);
B(1).FaceColor = [0.16 0.44 0.84];   % ECM (blue-ish)
B(2).FaceColor = [0.90 0.47 0.13];   % FNN (orange-ish);

ylabel('Average RMSE (mV)');
xlabel('Temperature');
xticks(temps_u);
xticklabels(arrayfun(@(T) sprintf('%d^{\\circ}C', T), temps_u, 'UniformOutput', false));
title('Temperature-wise average voltage RMSE: ECM vs FNN');

grid on; set(gca,'Box','on','Layer','top');
legend({'ECM (present)','FNN (reference)'}, 'Location','northwest','Box','off');

% 4) Extend Y-limits to make space for % labels
maxAvg = max([ecm_avg_rmse; fnn_avg_rmse], [], 'omitnan');
yl = ylim;
yl(1) = 0;
yl(2) = ceil(maxAvg*1.30);   % ~30% headroom
ylim(yl);
yspan = yl(2)-yl(1);
ypad  = 0.04 * yspan;

% 5) Add %Δ labels above each temperature group (centered on the group)
for t = 1:numel(temps_u)
    if isnan(dperc_temp(t)), continue; end

    % x-position at the center of the group (bar() uses temps_u directly)
    xpos = temps_u(t);

    % y-position just above the higher of the two bars
    ymax = max(ecm_avg_rmse(t), fnn_avg_rmse(t));
    ypos = ymax + ypad;

    % color: green if ECM better, red if worse
    col = [0.10 0.55 0.10];     % green
    if dperc_temp(t) >= 0
        col = [0.75 0.10 0.10]; % red
    end

    text(xpos, ypos, sprintf('%+.1f%%', dperc_temp(t)), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom', ...
        'FontWeight','bold', 'FontSize', 8, 'Color', col);
end
savefig(fullfile(outDir, 'ECM_vs_FNN_TempWiseAvg.fig'));

