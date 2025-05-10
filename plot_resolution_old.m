% Load data from .dat file
data = load('E:\SS-OCT\Codes\Slider\FFT_Real_Output_old.dat');
x = data(:, 1);  % Sample number
y = data(:, 2);  % Magnitude

% Calibration parameters
start_sample = 340;
end_sample = 660;
thickness_m = 0.00084;
num_samples = end_sample - start_sample;
spatial_resolution = thickness_m / num_samples;

% Convert x-axis to meters
x_meters = (x - start_sample) * spatial_resolution;

% --- Find true peak from raw signal ---
[max_val, idx_max] = max(y);
x_peak = x_meters(idx_max);

% --- Select fitting window around the peak ---
window = 0.0002;  % +/- 200 µm
fit_mask = abs(x_meters - x_peak) < window;
x_fit = x_meters(fit_mask);
y_fit = y(fit_mask);

% Light smoothing (optional)
y_fit_smooth = smooth(y_fit, 5);  % Small window

% --- Define Gaussian model ---
gaussEqn = @(b, x) b(1) * exp(-((x - b(2)).^2) / (2 * b(3)^2));

% Initial parameter guess: [amplitude, mean, stddev]
initial_guess = [max_val, x_peak, 0.00005];

% Fit using nonlinear least squares
opts = optimset('Display','off');
best_fit = lsqcurvefit(gaussEqn, initial_guess, x_fit, y_fit_smooth, [], [], opts);

% Generate fitted curve
y_fit_gauss = gaussEqn(best_fit, x_fit);

% --- Compute FWHM and Area under Gaussian ---
sigma = best_fit(3);
FWHM = 2.355 * sigma;
area = best_fit(1) * sigma * sqrt(2 * pi);

% --- Plot: Full Signal with Overlay ---
figure;

subplot(2,1,1);
plot(x_meters, y, 'b-', 'DisplayName', 'Raw Data');
hold on;
plot(x_fit, y_fit_gauss, 'r-', 'LineWidth', 2, 'DisplayName', 'Gaussian Fit');
xlabel('Depth (meters)');
ylabel('Amplitude');
title('Full Signal with Local Gaussian Fit');
legend;
grid on;

% --- Plot: Zoomed-In View ---
subplot(2,1,2);
plot(x_fit, y_fit, 'b-', 'DisplayName', 'Raw Data (Zoomed)');
hold on;
plot(x_fit, y_fit_gauss, 'r-', 'LineWidth', 2, 'DisplayName', 'Gaussian Fit');
xlabel('Depth (meters)');
ylabel('Amplitude');
title('Zoomed-In View Around Peak');
legend;
grid on;

% --- Prepare fit results text ---
fit_text = {
    sprintf('Amplitude: %.2f', best_fit(1)), ...
    sprintf('Center: %.2f µm', best_fit(2) * 1e6), ...
    sprintf('\\sigma: %.2f µm', sigma * 1e6), ...
    sprintf('FWHM: %.2f µm', FWHM * 1e6), ...
    sprintf('Area: %.2f', area)
};

% --- Display text box on zoomed-in plot ---
x_text = min(x_fit) + 0.5 * range(x_fit);
y_text = max(y_fit) * 0.95;  % Position just below peak

text(x_text, y_text, fit_text, ...
    'FontSize', 10, 'BackgroundColor', 'w', ...
    'EdgeColor', 'k', 'VerticalAlignment', 'top');

% --- Display Fit Results ---
fprintf('--- Gaussian Fit Parameters ---\n');
fprintf('Amplitude        : %.4f\n', best_fit(1));
fprintf('Center (µm)      : %.2f µm\n', best_fit(2) * 1e6);
fprintf('Std Dev (?, µm)  : %.2f µm\n', sigma * 1e6);
fprintf('FWHM             : %.2f µm\n', FWHM * 1e6);
fprintf('Area under Curve : %.4f\n', area);
