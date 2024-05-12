% Define folders
audioFolder = fullfile(pwd, 'audio');
spectrogramFolder = fullfile(pwd, 'spectrograms');

% Create the spectrogram folder if it doesn't exist
if ~exist(spectrogramFolder, 'dir')
    mkdir(spectrogramFolder);
end

% List all .wav files in the audio folder
audioFiles = dir(fullfile(audioFolder, '*.wav'));

% Parameters for spectrogram
window = 256;
noverlap = 0.75 * window;
nfft = 512;

% Process each audio file
for k = 1:length(audioFiles)
    % Full path to the current audio file
    audioFilePath = fullfile(audioFolder, audioFiles(k).name);
    
    % Read the audio file
    [signal, Fs] = audioread(audioFilePath);
    
    % Use the first channel if it's stereo
    if size(signal, 2) > 1
        signal = signal(:, 1);
    end
    
    % Generate the spectrogram
    figure('Visible', 'off'); % Create a figure for the spectrogram without displaying it
    spectrogram(signal, window, noverlap, nfft, Fs, 'yaxis');
    colormap('jet');
    
    % Make adjustments to remove axes, labels, and borders
    set(gca, 'Position', [0 0 1 1]); % Make the axes occupy the whole figure
    axis off; % Turn off the axis
    
    % Save the figure
    spectrogramFileName = fullfile(spectrogramFolder, [audioFiles(k).name, '.png']);
    exportgraphics(gca, spectrogramFileName, 'Resolution', 300);
    close(gcf); % Close the figure
end

disp('Spectrograms have been generated and saved.');
