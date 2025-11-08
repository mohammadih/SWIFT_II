function [signalOut, ofdmResponse, timingOffset, fs, tau_rms, Bc] = my_NRChannel_modified(signalIn, ch_type, OFDM_Response, cfgDL)
% my_NRChannel: Apply 3GPP CDL (HFS/MFS/FF) to IQ and/or return OFDM H(f,t)
% Inputs:
%   signalIn       : complex IQ from nrWaveformGenerator(cfgDL)
%   ch_type        : 'HFS' | 'MFS' | 'FF'
%   OFDM_Response  : true -> return OFDM H(f,t) on the NR grid
%   cfgDL (opt.)   : your nrDLCarrierConfig used to generate the IQ
%                    (if provided, numerology/sample rate are auto-aligned)
% Outputs:
%   signalOut      : IQ after channel (always produced)
%   ofdmResponse   : OFDM response if OFDM_Response=true
%   timingOffset   : timing offset reported by channel (OFDM mode)
%   fs             : channel sample rate actually used
%   tau_rms, Bc    : RMS delay spread and coherence bandwidth estimate

% --- defaults
signalOut = []; ofdmResponse = []; timingOffset = []; tau_rms = NaN; Bc = NaN;

% ---------- Numerology binding ----------
if nargin >= 4 && ~isempty(cfgDL)
    scs_khz   = cfgDL.SCSCarriers{1}.SubcarrierSpacing;   % 15
    NSizeGrid = cfgDL.SCSCarriers{1}.NSizeGrid;           % 159
    % If you want to probe only the BWP for H(f,t), swap the next line to:
    % NSizeGrid = cfgDL.BandwidthParts{1}.NSizeBWP;       % 8
else
    % fallback to your generator defaults
    scs_khz   = 15;
    NSizeGrid = 159;
end

carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = scs_khz;
carrier.NSizeGrid         = NSizeGrid;
infoGrid = nrOFDMInfo(carrier);
fs = infoGrid.SampleRate;                     % MUST match waveform numerology

% ---------- Base CDL channel ----------
cdl = nrCDLChannel;
cdl.DelayProfile = 'Custom';
cdl.CarrierFrequency = 3.5e9;                 % passband center for Doppler
cdl.SampleRate = fs;                          % lock to NR numerology
cdl.NormalizeChannelOutputs = true;
cdl.NormalizePathGains    = false;
cdl.TransmitAntennaArray.Size = [1 1 1 1 1];
cdl.ReceiveAntennaArray.Size  = [1 1 1 1 1];
cdl.MaximumDopplerShift = 5;                  % gentle time variation

% ---------- Scenario taps ----------
ch_type = upper(string(ch_type));
switch ch_type
    case "HFS"   % Highly frequency-selective
        cdl.PathDelays       = [0, 0.30e-6, 0.70e-6, 1.20e-6, 1.80e-6];
        cdl.AveragePathGains = [0, -2.5, -5, -7, -10] - 40;
        cdl.HasLOSCluster    = false;  % K-factor ignored when false
        L = numel(cdl.PathDelays);
        cdl.AnglesAoA = zeros(1,L);  cdl.AnglesAoD = zeros(1,L);
        cdl.AnglesZoA = 90*ones(1,L); cdl.AnglesZoD = 90*ones(1,L);
        desc = 'Highly Frequency Selective Channel';

    case "MFS"   % Moderately frequency-selective
        cdl.PathDelays       = [0, 30e-9, 65e-9, 110e-9, 160e-9];
        cdl.AveragePathGains = [0, -3, -6, -9, -12] - 40;
        cdl.HasLOSCluster    = true; cdl.KFactorFirstCluster = 3;
        L = numel(cdl.PathDelays);
        cdl.AnglesAoA = linspace(0,40,L);
        cdl.AnglesAoD = linspace(-10,10,L);
        cdl.AnglesZoA = 70*ones(1,L); cdl.AnglesZoD = 90*ones(1,L);
        desc = 'Moderately Frequency Selective Channel';

    case "FF"    % Frequency-flat (within band)
        cdl.PathDelays       = 0;
        cdl.AveragePathGains = -40;
        cdl.HasLOSCluster    = true; cdl.KFactorFirstCluster = 20;
        cdl.MaximumDopplerShift = 0;
        L = 1; cdl.AnglesAoA=0; cdl.AnglesAoD=0; cdl.AnglesZoA=90; cdl.AnglesZoD=90;
        desc = 'Frequency-Flat Channel';

    otherwise
        error('Unknown ch_type "%s". Use HFS/MFS/FF.', ch_type);
end

% ---------- Stats ----------
[tau_rms, Bc] = reportDelaySpread(cdl.PathDelays, cdl.AveragePathGains);
fprintf('[%s] tau_rms=%.1f ns, Bc≈%.2f MHz | SCS=%d kHz, N_RB=%d, fs=%.2f Msps\n', ...
    desc, tau_rms*1e9, Bc/1e6, scs_khz, NSizeGrid, fs/1e6);

% ---------- (A) OFDM response (optional) ----------
if OFDM_Response
    cdl.ChannelFiltering      = false;
    cdl.ChannelResponseOutput = 'ofdm-response';
    reset(cdl);
    [ofdmResponse, timingOffset] = cdl(carrier);        % [Nsc x Nsym x Nr x Nt]

    % Heatmap visualization
    H = abs(ofdmResponse(:,:,1,1)).';
    figure('Color','w'); imagesc(H); axis xy;
    xlabel('Subcarrier index'); ylabel('OFDM symbol index');
    title(sprintf('%s — |H(f,t)|, SCS=%d kHz, N_{RB}=%d', desc, scs_khz, NSizeGrid));
    colorbar; colormap turbo;
end

% ---------- (B) Filter the IQ (always) ----------
cdl.ChannelFiltering = true;
reset(cdl);
signalOut = cdl(signalIn);

end % function


% --- Local helper
function [tau_rms, Bc] = reportDelaySpread(pathDelays, pathGains_dB)
P = 10.^(pathGains_dB(:)/10); P = P/sum(P);
tau = pathDelays(:); tbar = sum(P.*tau);
tau_rms = sqrt(sum(P.*(tau-tbar).^2));
Bc = 1/(5*tau_rms);
end
