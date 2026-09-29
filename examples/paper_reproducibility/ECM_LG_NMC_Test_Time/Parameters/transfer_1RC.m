clc; clear; close all;

% ========= USER INPUTS =========
rootFolder = "E:\ITEC\LG_E66\Parameters\ITEC_2RC_Parameters_Convert_csv_to_mat";   % <-- main folder containing subfolders
% ===============================

% Find all CSV files inside rootFolder and all subfolders
csvFiles = dir(fullfile(rootFolder, "**", "*.csv"));

fprintf("Found %d CSV files.\n", numel(csvFiles));

for k = 1:numel(csvFiles)
    try
        % Full path of current CSV
        csvPath = fullfile(csvFiles(k).folder, csvFiles(k).name);

        % Output MAT file in same folder, same base name
        [~, baseName, ~] = fileparts(csvFiles(k).name);
        outMat = fullfile(csvFiles(k).folder, baseName + ".mat");

        % Read table
        T = readtable(csvPath);

        vars = lower(string(T.Properties.VariableNames));
        getcol = @(name) T{:, find(vars == lower(string(name)), 1, 'first')};

        % ---- read CSV columns ----
        SOC_HPPC = getcol("SOC_HPPC");
        OCV_HPPC = getcol("OCV_HPPC");

        R0   = getcol("R0");
        R1   = getcol("R1");
        R2   = getcol("R2");
        tau1 = getcol("tau1");
        tau2 = getcol("tau2");

        % ---- compute capacitances ----
        C1 = tau1 ./ R1;
        C2 = tau2 ./ R2;

        % ---- format exactly like workspace ----
        SOC_HPPC = SOC_HPPC(:).';   
        OCV_HPPC = OCV_HPPC(:).';   

        par_table_2RC = [R0(:), R1(:), C1(:), R2(:), C2(:)];

        % ---- save ONLY 3 variables ----
        save(outMat, "par_table_2RC", "SOC_HPPC", "OCV_HPPC", "-mat");

        fprintf("Saved: %s\n", outMat);

    catch ME
        fprintf("Error in file: %s\n", csvPath);
        fprintf("Reason: %s\n\n", ME.message);
    end
end

fprintf("Done.\n");