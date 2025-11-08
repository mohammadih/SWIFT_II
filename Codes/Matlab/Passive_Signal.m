function P_N = Passive_Signal(p,N)

% === CONSTANTS ===
K = 1.38e-23;             % Boltzmann constant (J/K)
fs = 30.72e6;             % Sampling rate [Hz]
mu_passive = 0;           % Mean of noise
% N = 30720;                % Number of samples (~1 ms duration)

% === INPUTS ===
% T = 290;                  % Temperature in Kelvin
P_dBm_desired = p;     % Desired power level in dBm

% === COMPUTE ===
% Convert desired power from dBm to Watts
P_watt_desired = 10^((P_dBm_desired - 30)/10);  % [W]

% Adjust the bandwidth to match sample spacing (fs/N)
B = fs;  % total sampling bandwidth (can be adjusted per application)

% Calculate thermal noise power at temperature T (reference only)

T = P_watt_desired/(K*B);

P_thermal_watt = K * T * B;

% Display actual thermal noise (not applied, just reference)
fprintf('Reference thermal noise @%gK: %.2f dBm\n', T, ...
    10*log10(P_thermal_watt) + 30);

% === NOISE SIGNAL GENERATION ===
% For complex signal, P = 2 * sigma^2  ⇒ sigma = sqrt(P/2)
sigma_passive = sqrt(P_watt_desired / 2);

% Generate complex Gaussian noise with specified power
P_N = normrnd(mu_passive, sigma_passive, [N,1]) + ...
                 1i * normrnd(mu_passive, sigma_passive, [N,1]);

% === VERIFICATION ===
% Compute measured power in dBm
P_meas_dBm = 10 * log10(mean(abs(P_N).^2)) + 30;
fprintf('Generated signal power: %.2f dBm\n', P_meas_dBm);

% visualize
% if exist('spectrumAnalyzer', 'class')
%     sa = spectrumAnalyzer('SampleRate', fs, ...
%         'Title', sprintf('Passive Signal at %d dBm (T=%.1fK)', P_dBm_desired, T), ...
%         'SpectrumType', 'power-density', ...
%         'ShowLegend', true);
%     sa(P_N);
% else
%     sa = [];
% end

persistent sa
if isempty(sa) || ~isvalid(sa)
    sa = spectrumAnalyzer('SampleRate', fs, ...
        'Title', sprintf('Passive Signal at %d dBm (T=%.1fK)', P_dBm_desired, T), ...
        'SpectrumType', 'power-density', ...
        'ShowLegend', true);
end
sa(P_N);


end