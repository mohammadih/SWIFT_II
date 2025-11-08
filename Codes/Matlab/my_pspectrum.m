function [p,f,t_pwr] = my_pspectrum(x,fs,type,fig_title)

% type = 'spectrogram'/ 'persistence' / 'power'

switch lower(type)
    case 'spectrogram'
        [p,f,t_pwr] = pspectrum(x,fs,type);

        figure
        waterfall(f, t_pwr, pow2db(p)');
        title(fig_title)
        xlabel('Frequency (Hz)')
        ylabel('Time (s)')
        zlabel('Power (dB)')
        % grid on

    case 'persistence'
        [p,f,t_pwr] = pspectrum(x,fs,type);
        figure
        pspectrum(x,fs,type);
        title(fig_title)
        % xlabel('Frequency')
        % ylabel('Power (dB)')
        % grid on

    case 'power'
        [p,f] = pspectrum(x,fs,type);
        t_pwr = [];
        figure
        plot(f, pow2db(p))
        title(fig_title)
        xlabel('Frequency')
        ylabel('Power (dB)')
        grid on

    otherwise
        error('Invalid type: choose spectrogram, persistence, or power.');

end

end