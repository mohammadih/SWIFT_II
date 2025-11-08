clear all; close all; clc;

mu_passive = 0;
sigma_passive = -90;  % consider lower noise levels for intruments
sigma_passive_linear = 10^(sigma_passive/10);
decimation_rate = 2;
nFeatures = 4;
fs = 30.72e6;
p_n = -120; % passive signal power

%% 10s data generation
% Define the file path
file_path1 = "J:\PC Shared-WD\Document\Datasets\SWIFT\April 2\OneDrive_1_4-14-2025\fb_as_8RBG_10dB_16QAM.dat";
file_path2 = "J:\PC Shared-WD\Document\Datasets\SWIFT\April 2\OneDrive_1_4-14-2025\fb_as_2RBG_10dB_16QAM.dat";

% Open the file for reading in binary mode
fileID1 = fopen(file_path1, 'rb');
fileID2 = fopen(file_path2, 'rb');
if fileID1 == -1
    error("Error opening file. Check if the file exists and you have read permissions.");
end

if fileID2 == -1
    error("Error opening file. Check if the file exists and you have read permissions.");
end
% Read data as floating point (assuming float32 format)
data1 = fread(fileID1, 'float32');
data2 = fread(fileID2, 'float32');

% Close file
fclose(fileID1);
fclose(fileID2);

data1 = data1(1:2:end) + 1i*data1(2:2:end);
data2 = data2(1:2:end) + 1i*data2(2:2:end);

% data = decimate(data,decimation_rate);

N = numel(data1);

% Dataset for test

% Define the file path
file_path1_test = "J:\PC Shared-WD\Document\Datasets\SWIFT\April 2\OneDrive_1_4-14-2025\fb_as_8RBG_0dB_16QAM.dat";
file_path2_test = "J:\PC Shared-WD\Document\Datasets\SWIFT\April 2\OneDrive_1_4-14-2025\fb_as_2RBG_0dB_16QAM.dat";

% Open the file for reading in binary mode
fileID1_test = fopen(file_path1_test, 'rb');
fileID2_test = fopen(file_path2_test, 'rb');
if fileID1_test == -1
    error("Error opening file. Check if the file exists and you have read permissions.");
end

if fileID2_test == -1
    error("Error opening file. Check if the file exists and you have read permissions.");
end
% Read data as floating point (assuming float32 format)
data1_test = fread(fileID1_test, 'float32');
data2_test = fread(fileID2_test, 'float32');

% Close file
fclose(fileID1_test);
fclose(fileID2_test);

data1_test = data1_test(1:2:end) + 1i*data1_test(2:2:end);
data2_test = data2_test(1:2:end) + 1i*data2_test(2:2:end);

%% 1s data generation
% load ref2_IQ20.mat
% load ref2_IQ80.mat

% I80 = t80(:,1);
% Q80 = t80(:,2);
% signal80 = I80 + 1i * Q80;  % Combine I and Q to form the complex signal
% upperlimit = max(find(abs(t80(:,2))>0));
% I20 = t20(:,1);
% Q20 = t20(:,2);
% signal20 = I20 + 1i * Q20;  % Combine I and Q to form the complex signal

% N = numel(signal80);


% my_NRChannel(signal In, OFDM_Response = true/false)
SignalIn = data1 + data2;
[signalOut,ofdmResponse,timingOffset,fs] = my_NRChannel(SignalIn,'HFS', true);
% fs_decimated = fs / decimation_rate;

% passive_signal = normrnd(mu_passive, sqrt(sigma_passive_linear),[N,1]) +...
%     1i*normrnd(mu_passive, sqrt(sigma_passive_linear),[N,1]);
passive_signal = Passive_Signal(p_n,N);

total_received_signal = passive_signal + signalOut;
% total_received_signal_dec = decimate(total_received_signal, decimation_rate);

% type = 'spectrogram'/ 'persistence' / 'power'
[p_spect,f_spect] = my_pspectrum(total_received_signal,fs,'power','Signal Spectrum');

% [p_spect,f_spect] = my_pspectrum(data2,fs,'power','Signal Spectrum');

% Create interleaved I/Q data
% IQ_interleaved = zeros(2 * length(total_received_signal), 1, 'single');
% IQ_interleaved(1:2:end) = real(total_received_signal);  % I
% IQ_interleaved(2:2:end) = imag(total_received_signal);  % Q

% Define full path and filename
% output_path = "J:\PC Shared-WD\Document\Datasets\SWIFT\April 2\OneDrive_1_4-14-2025\RX Side\rx_8rbg_10dB_16QAM.dat";

% Open file and write
% fileID_Out = fopen(output_path, 'wb');
% fwrite(fileID_Out, IQ_interleaved, 'float32');
% fclose(fileID_Out);
% 
% disp("IQ samples saved successfully!");



% Open file and write
% fileID_In_test = fopen(output_path,"rb");
% data_In_test = fread(fileID_In_test,'float32');
% fclose(fileID_In_test);
% 
% dataID_In_test = data_In_test(1:2:end) + 1i*data_In_test(2:2:end);


% rTotal_rsp_2Type = reshape(total_received_signal,[],nFeatures);
% rTotal_batch_2Type = [real(rTotal_rsp_2Type),imag(rTotal_rsp_2Type)];
% fft_resultITr_2Type = fftshift(fft(total_received_signal));
% power_spectrum_totalTr_2Type = abs(fft_resultITr_2Type).^2 /(N*fs); % Normalize by N^2 for correct energy scaling
% power_spectrum_totalTr_2Type = power_spectrum_totalTr_2Type*fs/N; % Scale by frequency resolution
% magnitude_spectrumTr_2Type = 10*log10(abs(power_spectrum_totalTr_2Type));
% magnitude_spectrumTr_rsp_2Type = reshape(magnitude_spectrumTr_2Type,[],nFeatures);

% Region Selection
% Red box subcarrier extraction
% f_low = 5.5e6;
% f_high = 6.8e6;
% N = length(total_received_signal);
% freqs = linspace(-fs/2, fs/2, N);
% spectrum = fftshift(fft(total_received_signal));
% mask = (freqs >= f_low) & (freqs <= f_high);
% filtered_spectrum = zeros(size(spectrum));
% filtered_spectrum(mask) = spectrum(mask);
% filtered_signal = ifft(ifftshift(filtered_spectrum), 'symmetric');

% Format for training
% rFiltered = reshape(filtered_signal, [], nFeatures);
% Input810_batch = [real(rFiltered), imag(rFiltered)];


fft_resultITr = fftshift(fft(total_received_signal));
power_spectrum_totalTr = abs(fft_resultITr).^2 /(N*fs); % Normalize by N^2 for correct energy scaling
power_spectrum_totalTr = power_spectrum_totalTr*fs/N; % Scale by frequency resolution
magnitude_spectrumTr = 10*log10(abs(power_spectrum_totalTr));
magnitude_spectrumTr_rsp = reshape(magnitude_spectrumTr,[],nFeatures);

InputSourcesTr_rsp = reshape(SignalIn,[],nFeatures);
InputSourcesTr_2Type_batch_label = [real(InputSourcesTr_rsp),imag(InputSourcesTr_rsp)];

InputSourcesTr_rsp = reshape(total_received_signal,[],nFeatures);
InputSourcesTr_2Type_batch = [real(InputSourcesTr_rsp),imag(InputSourcesTr_rsp)];

[net, tr] = SIC1_DL_Reg([InputSourcesTr_2Type_batch,magnitude_spectrumTr_rsp], ...
    InputSourcesTr_2Type_batch_label,nFeatures);

signalIn_test = data1_test + data2_test;
[signalOut_test,ofdmResponse_test,timingOffset_test,fs] = my_NRChannel(signalIn_test, false);
% passive_signal_test = normrnd(mu_passive, sqrt(sigma_passive_linear),[N,1]) +...
%     1i*normrnd(mu_passive, sqrt(sigma_passive_linear),[N,1]);
passive_signal_test = Passive_Signal(p_n,N);
total_received_signal_test = passive_signal_test + signalOut_test;

InputSourcesTr_rsp_test = reshape(total_received_signal_test,[],nFeatures);
InputSourcesTr_2Type_batch_test = [real(InputSourcesTr_rsp_test),imag(InputSourcesTr_rsp_test)];

fft_resultITst = fftshift(fft(total_received_signal_test));
power_spectrum_totalTst = abs(fft_resultITst).^2 /(N*fs); % Normalize by N^2 for correct energy scaling
power_spectrum_totalTst = power_spectrum_totalTst*fs/N; % Scale by frequency resolution
magnitude_spectrumTst = 10*log10(abs(power_spectrum_totalTst));
magnitude_spectrumTst_rsp = reshape(magnitude_spectrumTst,[],nFeatures);

yy = net([InputSourcesTr_2Type_batch_test,magnitude_spectrumTst_rsp]');
yy = yy';
rr_filtered1_IQ = yy(:,1:nFeatures) + 1i*yy(:,nFeatures+1:end);
rr_filtered1_IQ = rr_filtered1_IQ(:);


SIC1 = total_received_signal - rr_filtered1_IQ;
[p_spect_filtered,f_spect_filtered] = my_pspectrum(rr_filtered1_IQ,fs,'power','Reconstructes Signal Spectrum');
[p_spect_recon,f_spect_recon] = my_pspectrum(SIC1,fs,'power','Filtered Signal Spectrum');

figure
subplot(2,1,1)
pspectrum(total_received_signal(1:upperlimit),fs,'spectrogram','MinThreshold',-100)
title('Received Signal')
subplot(2,1,2)
pspectrum(rr_filtered1_IQ(1:upperlimit),fs,'spectrogram','MinThreshold',-100)
title('Filtered Signal ')

% [p, f] = pspectrum(total_received_signal, fs, "power"); % 1 kHz resolution
% figure;
% plot(f, pow2db(p));
% title('Power Spectrum (Zoomed)');
% xlabel('Frequency (Hz)');
% ylabel('Power (dB)');
% % ylim([-130 -70]); % Manually set y-axis
% grid on;
% 
% 
% [pxx,f] = pwelch(ref2_IQ80,4096,2048,4096,fs,"centered");
% 
% figure
% plot(f,pow2db(pxx))
% 
% waterfall(f,t,p')
% xlabel('Frequency (Hz)')
% ylabel('Time (seconds)')
% wtf = gca;
% wtf.XDir = 'reverse';
% view([30 45])