clc; clear; close all;

%% ========================================================================
%  1RC ECM — 6-TEMPERATURE TEST + 1000-HOUR INFERENCE BENCHMARK
%
%  Temperatures:
%      -20, -10, 0, 10, 25, 40 degC
%
%  Based on Highest_Final_RC_With_Temp.m
%
%  PART A
%  ------
%  - Evaluates ALL matching test CSVs at ALL six temperatures.
%  - Times ONLY the 1RC model inference call.
%  - Reports inference time per file, per temperature, and overall.
%
%  PART B
%  ------
%  - Builds exactly 1000 hours of contiguous 1-second test inputs from
%    all available test files across all six temperatures.
%  - Repeats the complete source-file sequence as needed to reach 1000 h.
%  - Makes ONE continuous 1RC model call over 3,600,000 samples.
%  - Reports model-only time and seconds per hour of test data.
%
%  For fair model comparison, run this on the SAME PC used for the other
%  models, under comparable background-process conditions.
%
%  PARAMETER MAT FILE REQUIREMENTS
%  -------------------------------
%  Each MAT file must contain:
%    - 15 x 3 numeric parameter matrix: [R0 R1 C1]
%    - 15 x 1 SOC breakpoint vector
%    - 15 x 1 OCV vector
%
%  TEST CSV REQUIRED COLUMNS
%  -------------------------
%    Adjusted_Test_Time_s_
%    Voltage_V_
%    Current_A_
%    SOC
%    Battery_Temp_degC
%
%  SOC can be either 0..1 or 0..100.
% ========================================================================


%% =========================== USER SETTINGS ===============================

% Six characterized 1RC parameter files
paramFiles = { ...
    "Parameters/ITEC_1RC_Parameters_Convert_csv_to_mat/Highest/par_optimized_n20_1RC_perstep_pybop.mat", ...
    "Parameters/ITEC_1RC_Parameters_Convert_csv_to_mat/Highest/par_optimized_n10_1RC_perstep_pybop.mat", ...
    "Parameters/ITEC_1RC_Parameters_Convert_csv_to_mat/Highest/par_optimized_0_1RC_perstep_pybop.mat", ...
    "Parameters/ITEC_1RC_Parameters_Convert_csv_to_mat/Highest/par_optimized_10_1RC_perstep_pybop.mat", ...
    "Parameters/ITEC_1RC_Parameters_Convert_csv_to_mat/Highest/par_optimized_25_1RC_perstep_pybop.mat", ...
    "Parameters/ITEC_1RC_Parameters_Convert_csv_to_mat/Highest/par_optimized_40_1RC_perstep_pybop.mat" ...
};

fileTempMap = containers.Map( ...
    ["n20","n10","0","10","25","40"], ...
    [-20,-10,0,10,25,40] ...
);

testTemps = [-20,-10,0,10,25,40];

% Exact folder names if they follow the same pattern as your existing
% "0C_DCH_Only" folder. If any name differs, edit that line only.
explicitTestFolders = containers.Map('KeyType','double','ValueType','char');
explicitTestFolders(-20) = 'n20C_DCH_Only';
explicitTestFolders(-10) = 'n10C_DCH_Only';
explicitTestFolders(0)   = '0C_DCH_Only';
explicitTestFolders(10)  = '10C_DCH_Only';
explicitTestFolders(25)  = '25C_DCH_Only';
explicitTestFolders(40)  = '40C_DCH_Only';

% ALL matching CSVs are used; the script does not stop after the first match.
keywords = ["UDDS","HWFET","LA92","US06"];

% Dr. Phil-style long inference benchmark
BENCHMARK_HOURS = 1000;
BENCHMARK_DT_S  = 1.0;
BENCHMARK_SAMPLES = round(BENCHMARK_HOURS*3600/BENCHMARK_DT_S);

% 1 = exact single 1000-h benchmark.
% Set to 3 or 5 later if you want mean/std across repeated long runs.
N_BENCHMARK_REPEATS = 1;

outputRoot = "Results\All_1RC_6Temps_Inference";
if ~exist(outputRoot,"dir"), mkdir(outputRoot); end


%% ======================= ENVIRONMENT RECORD ==============================

fprintf("\n============================================================\n");
fprintf("1RC ECM 6-TEMPERATURE INFERENCE BENCHMARK\n");
fprintf("============================================================\n");
fprintf("Computer name : %s\n", getenv('COMPUTERNAME'));
fprintf("MATLAB        : %s\n", version);
fprintf("Temperatures  : ");
fprintf("%g ", testTemps);
fprintf("degC\n");
fprintf("Benchmark     : %.0f h contiguous data @ %.1f s/sample\n", ...
    BENCHMARK_HOURS, BENCHMARK_DT_S);
fprintf("Samples       : %d\n", BENCHMARK_SAMPLES);
fprintf("============================================================\n\n");


%% ======================== ECM INITIALIZATION =============================
% Setup is timed separately from inference.

setupTic = tic;

nT = numel(paramFiles);
tempGrid = zeros(1,nT);

parCells = cell(1,nT);
ocvCells = cell(1,nT);
SOC_master = [];

for j = 1:nT
    f = string(paramFiles{j});

    if ~isfile(f)
        error("Parameter file not found: %s", f);
    end

    S = load(f);
    [par_table, soc_bp, ocv_bp] = extractParSOC_OCV_1RC(S);

    [soc_bp_s,ord] = sort(soc_bp(:),'ascend');
    par_table_s = par_table(ord,:);
    ocv_bp_s = ocv_bp(ord);

    % Preserve the original script's master-SOC-grid approach.
    if isempty(SOC_master)
        SOC_master = soc_bp_s(:);
    end

    par_on_master = interp1( ...
        soc_bp_s, par_table_s, SOC_master, 'linear','extrap');
    par_on_master = max(par_on_master,eps);

    ocv_on_master = interp1( ...
        soc_bp_s, ocv_bp_s, SOC_master, 'linear','extrap');

    parCells{j} = par_on_master;
    ocvCells{j} = ocv_on_master;

    tok = inferTempTokenFromFilename(f);
    if ~isKey(fileTempMap,tok)
        error("Could not map token '%s' to temperature for file: %s",tok,f);
    end
    tempGrid(j) = fileTempMap(tok);
end

[tempGrid,orderT] = sort(tempGrid,'ascend');
parCells = parCells(orderT);
ocvCells = ocvCells(orderT);

socGrid = SOC_master(:);
nSOC = numel(socGrid);

R0_grid  = zeros(nSOC,nT);
R1_grid  = zeros(nSOC,nT);
C1_grid  = zeros(nSOC,nT);
OCV_grid = zeros(nSOC,nT);

for j = 1:nT
    par = parCells{j};
    R0_grid(:,j)  = par(:,1);
    R1_grid(:,j)  = par(:,2);
    C1_grid(:,j)  = par(:,3);
    OCV_grid(:,j) = ocvCells{j};
end

F_R0  = griddedInterpolant({socGrid,tempGrid},R0_grid,'linear','linear');
F_R1  = griddedInterpolant({socGrid,tempGrid},R1_grid,'linear','linear');
F_OCV = griddedInterpolant({socGrid,tempGrid},OCV_grid,'linear','linear');

C1_grid_safe = max(C1_grid,realmin);
F_logC1 = griddedInterpolant( ...
    {socGrid,tempGrid},log(C1_grid_safe),'linear','linear');

ECM_setup_time_s = toc(setupTic);

fprintf("ECM setup time = %.6f s\n",ECM_setup_time_s);
fprintf("Loaded temperature grid: ");
fprintf("%g ",tempGrid);
fprintf("degC\n\n");


%% ===================== DISCOVER ALL TEST FILES ===========================

TestFiles = struct( ...
    'NominalTemp_C',{}, ...
    'Folder',{}, ...
    'FullPath',{}, ...
    'Name',{} ...
);

fprintf("===== TEST FILE DISCOVERY =====\n");

for it = 1:numel(testTemps)
    Tnom = testTemps(it);

    explicitFolder = string(explicitTestFolders(Tnom));
    folderPath = resolveTestFolder(Tnom,explicitFolder);

    allFiles = dir(fullfile(folderPath,"*.csv"));

    selected = false(numel(allFiles),1);
    for j = 1:numel(allFiles)
        nm = string(allFiles(j).name);
        selected(j) = any(contains(nm,keywords,'IgnoreCase',true));
    end

    theseFiles = allFiles(selected);

    if isempty(theseFiles)
        error("No matching test CSVs found in folder: %s",folderPath);
    end

    fprintf("%+g C: %s -> %d matching files\n", ...
        Tnom,folderPath,numel(theseFiles));

    for j = 1:numel(theseFiles)
        TestFiles(end+1).NominalTemp_C = Tnom; %#ok<SAGROW>
        TestFiles(end).Folder = string(folderPath);
        TestFiles(end).FullPath = string(fullfile(folderPath,theseFiles(j).name));
        TestFiles(end).Name = string(theseFiles(j).name);
    end
end

fprintf("TOTAL test files = %d\n\n",numel(TestFiles));


%% ========================================================================
% PART A — ALL REAL TEST FILES, ALL SIX TEMPERATURES
% ========================================================================

fprintf("\n============================================================\n");
fprintf("PART A: ALL TEST FILES, ALL SIX TEMPERATURES\n");
fprintf("============================================================\n");

PerFile = table();

% Retain model input sequences for the 1000-h benchmark.
SourceSOC  = {};
SourceTemp = {};
SourceI    = {};
SourceTime = {};

for k = 1:numel(TestFiles)

    Tnom = TestFiles(k).NominalTemp_C;
    fp = TestFiles(k).FullPath;

    data = readRequiredTestColumns(fp);

    t_raw = double(data.Adjusted_Test_Time_s_(:));
    V_raw = double(data.Voltage_V_(:));
    I_raw = double(data.Current_A_(:));

    soc_col = double(data.SOC(:));
    if max(soc_col,[],'omitnan') > 1.5
        SOCraw = soc_col;
    else
        SOCraw = 100*soc_col;
    end

    Tcell_raw = double(data.Battery_Temp_degC(:));

    good = isfinite(t_raw) & isfinite(V_raw) & isfinite(I_raw) & ...
           isfinite(SOCraw) & isfinite(Tcell_raw);

    t_raw = t_raw(good);
    V_raw = V_raw(good);
    I_raw = I_raw(good);
    SOCraw = SOCraw(good);
    Tcell_raw = Tcell_raw(good);

    [t_u,ia] = unique(t_raw,'stable');
    V_u   = V_raw(ia);
    I_u   = I_raw(ia);
    SOCu  = SOCraw(ia);
    Tc_u  = Tcell_raw(ia);

    if numel(t_u) < 2
        error("Not enough valid samples in %s",fp);
    end

    % Preserve the original evaluation: resample every real test to 1 s.
    T_meas   = (t_u(1):1:t_u(end)).';
    V_meas   = interp1(t_u,V_u,  T_meas,'linear','extrap');
    SOC_meas = interp1(t_u,SOCu, T_meas,'linear','extrap');
    I_meas   = interp1(t_u,I_u,  T_meas,'linear','extrap');
    Tcell    = interp1(t_u,Tc_u, T_meas,'linear','extrap');

    SOC_meas = min(max(SOC_meas,min(socGrid)),max(socGrid));
    Tcell    = min(max(Tcell,min(tempGrid)),max(tempGrid));

    % ---------------- MODEL-ONLY INFERENCE TIMER ----------------
    inferTic = tic;

    V_estim = model_OCV_R_1RC_SOC_TEMP_stable( ...
        SOC_meas,Tcell,I_meas,T_meas, ...
        F_R0,F_R1,F_logC1,F_OCV,socGrid,tempGrid);

    inference_s = toc(inferTic);
    % ------------------------------------------------------------

    err_V = V_estim - V_meas;
    err_mV = 1000*err_V;

    RMSE_mV = sqrt(mean(err_mV.^2,'omitnan'));
    MaxErr_mV = max(abs(err_mV),[],'omitnan');
    Bias_mV = 1000*mean(err_V,'omitnan');

    V_estim_noBias = V_estim - mean(err_V,'omitnan');
    RMSE_noBias_mV = sqrt(mean((1000*(V_estim_noBias-V_meas)).^2,'omitnan'));

    % 1-s resampling means numel(T_meas) samples represent this duration.
    duration_s = T_meas(end)-T_meas(1);
    duration_h = duration_s/3600;

    if duration_h > 0
        sec_per_test_h = inference_s/duration_h;
    else
        sec_per_test_h = NaN;
    end

    fprintf('%+4g C | %-42s | data=%8.4f h | infer=%9.6f s | %9.6f s/test-h | RMSE=%7.2f mV\n', ...
        Tnom,char(TestFiles(k).Name),duration_h,inference_s, ...
        sec_per_test_h,RMSE_mV);

    newRow = table( ...
        Tnom,TestFiles(k).Name,duration_h,numel(T_meas), ...
        inference_s,sec_per_test_h,RMSE_mV,Bias_mV, ...
        RMSE_noBias_mV,MaxErr_mV, ...
        'VariableNames',{ ...
        'NominalTemp_C','File','TestData_hours','Samples', ...
        'InferenceTime_s','Inference_s_per_test_hour', ...
        'RMSE_mV','Bias_mV','RMSE_noBias_mV','MaxAbsErr_mV'});

    if isempty(PerFile)
        PerFile = newRow;
    else
        PerFile = [PerFile;newRow]; %#ok<AGROW>
    end

    SourceSOC{end+1}  = SOC_meas; %#ok<SAGROW>
    SourceTemp{end+1} = Tcell; %#ok<SAGROW>
    SourceI{end+1}    = I_meas; %#ok<SAGROW>
    SourceTime{end+1} = T_meas; %#ok<SAGROW>
end

writetable(PerFile,fullfile(outputRoot,"PerFile_Inference_Timing.csv"));


%% ==================== PER-TEMPERATURE SUMMARY ===========================

PerTemp = table();

for it = 1:numel(testTemps)
    Tnom = testTemps(it);
    mask = PerFile.NominalTemp_C == Tnom;

    nFiles = sum(mask);
    data_h = sum(PerFile.TestData_hours(mask));
    infer_s = sum(PerFile.InferenceTime_s(mask));

    if data_h > 0
        sec_per_h = infer_s/data_h;
    else
        sec_per_h = NaN;
    end

    row = table( ...
        Tnom,nFiles,data_h,infer_s,sec_per_h, ...
        'VariableNames',{ ...
        'Temperature_C','NumFiles','TotalTestData_hours', ...
        'TotalInferenceTime_s','Inference_s_per_test_hour'});

    if isempty(PerTemp)
        PerTemp = row;
    else
        PerTemp = [PerTemp;row]; %#ok<AGROW>
    end
end

total_test_h = sum(PerFile.TestData_hours);
total_infer_s = sum(PerFile.InferenceTime_s);
overall_sec_per_h = total_infer_s/total_test_h;

writetable(PerTemp,fullfile(outputRoot,"PerTemperature_Inference_Timing.csv"));

fprintf("\n================ PART A SUMMARY ================\n");
disp(PerTemp);

fprintf("ALL SIX TEMPERATURES COMBINED:\n");
fprintf("  Number of test files        = %d\n",height(PerFile));
fprintf("  Total test-data duration    = %.6f h\n",total_test_h);
fprintf("  TOTAL MODEL INFERENCE TIME  = %.6f s\n",total_infer_s);
fprintf("  Average inference / test h  = %.9f s/h\n",overall_sec_per_h);
fprintf("================================================\n\n");


%% ========================================================================
% PART B — 1000-HOUR CONTIGUOUS INFERENCE BENCHMARK
% ========================================================================
%
% Data reading and 1000-h construction are OUTSIDE the inference timer.
% The model is initialized once, then receives one continuous 3.6M-sample
% call. No warm-up is used, so first-call/JIT overhead is naturally included
% in the long benchmark and becomes negligible over 1000 h.
% ========================================================================

fprintf("\n============================================================\n");
fprintf("PART B: BUILDING %.0f HOURS OF CONTIGUOUS TEST DATA\n",BENCHMARK_HOURS);
fprintf("============================================================\n");

% Part A data are already at 1-second sampling.
total_source_samples = sum(cellfun(@numel,SourceSOC));

fprintf("Unique source data available: %.4f h\n", ...
    total_source_samples*BENCHMARK_DT_S/3600);

N = BENCHMARK_SAMPLES;

SOC1000  = zeros(N,1);
TEMP1000 = zeros(N,1);
I1000    = zeros(N,1);

pos = 1;
cycleCount = 0;

while pos <= N
    cycleCount = cycleCount+1;

    for k = 1:numel(SourceSOC)
        nAvail = numel(SourceSOC{k});
        nTake = min(nAvail,N-pos+1);

        idxOut = pos:(pos+nTake-1);

        SOC1000(idxOut)  = SourceSOC{k}(1:nTake);
        TEMP1000(idxOut) = SourceTemp{k}(1:nTake);
        I1000(idxOut)    = SourceI{k}(1:nTake);

        pos = pos+nTake;

        if pos > N
            break;
        end
    end
end

TIME1000 = (0:N-1).'*BENCHMARK_DT_S;

fprintf("Constructed exactly %.3f h (%d samples).\n", ...
    N*BENCHMARK_DT_S/3600,N);
fprintf("Complete source sequence repeated %d time(s) to reach target.\n", ...
    cycleCount);

% Reduce memory pressure before the timed call.
clear SourceSOC SourceTemp SourceI SourceTime;


%% ==================== OFFICIAL 1000-HOUR CALL ===========================

benchTimes = zeros(N_BENCHMARK_REPEATS,1);

fprintf("\n===== OFFICIAL 1000-HOUR BENCHMARK =====\n");

for r = 1:N_BENCHMARK_REPEATS

    drawnow;

    benchTic = tic;

    V_1000 = model_OCV_R_1RC_SOC_TEMP_stable( ...
        SOC1000,TEMP1000,I1000,TIME1000, ...
        F_R0,F_R1,F_logC1,F_OCV,socGrid,tempGrid);

    benchTimes(r) = toc(benchTic);

    fprintf("Repeat %d/%d: %.6f s total = %.9f s per test-data hour\n", ...
        r,N_BENCHMARK_REPEATS,benchTimes(r), ...
        benchTimes(r)/BENCHMARK_HOURS);

    if r < N_BENCHMARK_REPEATS
        clear V_1000;
    end
end

meanBench_s = mean(benchTimes);
stdBench_s = std(benchTimes);
meanSecPerHour = meanBench_s/BENCHMARK_HOURS;

setupPlusInference_s = ECM_setup_time_s+meanBench_s;
setupPlusInference_perHour_s = setupPlusInference_s/BENCHMARK_HOURS;


%% ========================== FINAL REPORT ================================

fprintf("\n\n============================================================\n");
fprintf("FINAL 1RC 6-TEMPERATURE INFERENCE-TIME REPORT\n");
fprintf("============================================================\n");

fprintf("\nPART A — REAL TEST FILES\n");
fprintf("Temperatures                    : ");
fprintf("%g ",testTemps);
fprintf("C\n");
fprintf("Number of test files            : %d\n",height(PerFile));
fprintf("Total available test-data hours : %.6f h\n",total_test_h);
fprintf("Total inference time             : %.6f s\n",total_infer_s);
fprintf("Inference time per test hour     : %.9f s/h\n",overall_sec_per_h);

fprintf("\nPART B — 1000-HOUR CONTIGUOUS BENCHMARK\n");
fprintf("ECM one-time setup               : %.6f s\n",ECM_setup_time_s);
fprintf("Benchmark test-data length       : %.0f h\n",BENCHMARK_HOURS);
fprintf("Benchmark samples                : %d\n",N);
fprintf("Model-only inference mean        : %.6f s\n",meanBench_s);
fprintf("Model-only inference std         : %.6f s\n",stdBench_s);
fprintf("MODEL-ONLY TIME PER TEST HOUR    : %.9f s/h\n",meanSecPerHour);
fprintf("Setup + inference                : %.6f s\n",setupPlusInference_s);
fprintf("SETUP+INFERENCE PER TEST HOUR    : %.9f s/h\n", ...
    setupPlusInference_perHour_s);

fprintf("============================================================\n");


%% ============================ SAVE ======================================

BenchmarkSummary = table( ...
    string(getenv('COMPUTERNAME')),string(version), ...
    ECM_setup_time_s,height(PerFile),total_test_h,total_infer_s, ...
    overall_sec_per_h,BENCHMARK_HOURS,N,N_BENCHMARK_REPEATS, ...
    meanBench_s,stdBench_s,meanSecPerHour, ...
    setupPlusInference_s,setupPlusInference_perHour_s, ...
    'VariableNames',{ ...
    'Computer','MATLABVersion','ECM_SetupTime_s', ...
    'NumRealTestFiles','RealTestData_hours','RealTotalInference_s', ...
    'RealInference_s_per_test_hour', ...
    'Benchmark_hours','Benchmark_samples','Benchmark_repeats', ...
    'Benchmark_ModelOnly_mean_s','Benchmark_ModelOnly_std_s', ...
    'Benchmark_ModelOnly_s_per_test_hour', ...
    'Benchmark_SetupPlusInference_s', ...
    'Benchmark_SetupPlusInference_s_per_test_hour'});

writetable(BenchmarkSummary, ...
    fullfile(outputRoot,"FINAL_1RC_6Temps_Inference_Benchmark_Summary.csv"));

writetable(PerFile, ...
    fullfile(outputRoot,"PerFile_Inference_Timing.csv"));

writetable(PerTemp, ...
    fullfile(outputRoot,"PerTemperature_Inference_Timing.csv"));

fprintf("\nSaved timing results to:\n");
fprintf("  %s\n",fullfile(outputRoot, ...
    "FINAL_1RC_6Temps_Inference_Benchmark_Summary.csv"));
fprintf("  %s\n",fullfile(outputRoot,"PerFile_Inference_Timing.csv"));
fprintf("  %s\n",fullfile(outputRoot,"PerTemperature_Inference_Timing.csv"));


%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function data = readRequiredTestColumns(fp)

    required = { ...
        'Adjusted_Test_Time_s_', ...
        'Voltage_V_', ...
        'Current_A_', ...
        'SOC', ...
        'Battery_Temp_degC' ...
    };

    opts = detectImportOptions(fp,'VariableNamingRule','modify');

    available = opts.VariableNames;
    missing = required(~ismember(required,available));

    if ~isempty(missing)
        error("Missing required column(s) in %s: %s", ...
            fp,strjoin(string(missing),", "));
    end

    opts.SelectedVariableNames = required;
    data = readtable(fp,opts);
end


function folderPath = resolveTestFolder(Tnom,explicitFolder)

    if strlength(explicitFolder) > 0 && isfolder(explicitFolder)
        folderPath = explicitFolder;
        return;
    end

    % If the configured folder is absent, try common alternatives.
    if Tnom < 0
        a = abs(Tnom);
        candidates = [ ...
            string(sprintf("n%dC_DCH_Only",a)), ...
            string(sprintf("-%dC_DCH_Only",a)), ...
            string(sprintf("N%dC_DCH_Only",a)), ...
            string(sprintf("n%d_DCH_Only",a)), ...
            string(sprintf("-%d_DCH_Only",a)) ...
        ];
    else
        candidates = [ ...
            string(sprintf("%dC_DCH_Only",Tnom)), ...
            string(sprintf("%d_DCH_Only",Tnom)) ...
        ];
    end

    for i = 1:numel(candidates)
        if isfolder(candidates(i))
            folderPath = candidates(i);
            return;
        end
    end

    error([ ...
        "Could not find the test folder for %+g C.\n" ...
        "Set explicitTestFolders(%g) to the exact folder path."], ...
        Tnom,Tnom);
end


function V = model_OCV_R_1RC_SOC_TEMP_stable( ...
    SOC,Tcell,current,time, ...
    F_R0,F_R1,F_logC1,F_OCV,socGrid,tempGrid)

    SOC = SOC(:);
    current = current(:);
    time = time(:);

    if isscalar(Tcell)
        Tq = repmat(Tcell,size(SOC));
    else
        Tq = Tcell(:);
    end

    SOCq = min(max(SOC,min(socGrid)),max(socGrid));
    Tq   = min(max(Tq,min(tempGrid)),max(tempGrid));

    R0  = F_R0(SOCq,Tq);
    R1  = F_R1(SOCq,Tq);
    C1  = exp(F_logC1(SOCq,Tq));
    OCV = F_OCV(SOCq,Tq);

    V_R0 = R0.*current;

    t = time(:);
    i = current(:);
    dt = [0;diff(t)];

    Vrc1 = zeros(size(t));

    for k = 2:numel(t)
        dtk = dt(k);

        if ~(isfinite(dtk) && dtk > 0)
            dtk = 1.0;
        end

        R1k = max(R1(k-1),eps);
        C1k = max(C1(k-1),eps);

        a1 = exp(-dtk/max(R1k*C1k,1e-12));

        Vrc1(k) = ...
            a1*Vrc1(k-1)+R1k*(1-a1)*i(k-1);
    end

    V = OCV+V_R0+Vrc1;
end


function tok = inferTempTokenFromFilename(f)

    f = char(f);

    if contains(f,'n20'), tok = "n20"; return; end
    if contains(f,'n10'), tok = "n10"; return; end

    if contains(f,'_0_'),  tok = "0";  return; end
    if contains(f,'_10_'), tok = "10"; return; end
    if contains(f,'_25_'), tok = "25"; return; end
    if contains(f,'_40_'), tok = "40"; return; end

    m = regexp(f,'par_optimized_([n]?\d+)_','tokens','once');

    if ~isempty(m)
        tok = string(m{1});
        return;
    end

    error("Could not infer temperature token from filename: %s",f);
end


function [par_table,soc_bp,ocv_bp] = extractParSOC_OCV_1RC(S)

    par_table = [];
    soc_bp = [];
    ocv_bp = [];

    fn = fieldnames(S);

    % Parameter table: 15 x 3
    for i = 1:numel(fn)
        v = S.(fn{i});
        if isnumeric(v) && ismatrix(v) && all(size(v)==[15 3])
            par_table = v;
            break;
        end
    end

    if isempty(par_table)
        error("No 15x3 [R0 R1 C1] parameter table found in MAT file.");
    end

    % SOC vector: length 15, prefer field name containing "soc"
    for i = 1:numel(fn)
        v = S.(fn{i});
        if isnumeric(v) && isvector(v) && numel(v)==15 && ...
                contains(lower(fn{i}),'soc')
            soc_bp = v(:);
            break;
        end
    end

    if isempty(soc_bp)
        for i = 1:numel(fn)
            v = S.(fn{i});
            if isnumeric(v) && isvector(v) && numel(v)==15
                soc_bp = v(:);
                break;
            end
        end
    end

    if isempty(soc_bp)
        error("No SOC breakpoint vector (length 15) found in MAT file.");
    end

    % OCV vector: length 15, prefer field name containing "ocv"
    for i = 1:numel(fn)
        v = S.(fn{i});
        if isnumeric(v) && isvector(v) && numel(v)==15 && ...
                contains(lower(fn{i}),'ocv')
            ocv_bp = v(:);
            break;
        end
    end

    if isempty(ocv_bp)
        error("No OCV breakpoint vector (length 15) found in MAT file.");
    end
end
