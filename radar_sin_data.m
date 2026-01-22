clear; clc; close all;

%% 1. 파라미터 설정
num_samples = 5000;    % 데이터 개수 (5,000개)
signal_len = 1024;     % 신호 길이
fs = 100e6;            % 샘플링 주파수 (100 MHz)
t = linspace(0, 10e-6, signal_len); % 시간 축 (0 ~ 10us)

% 목표 주파수 (예: 5 MHz)
target_freq = 5e6; 

% 데이터를 담을 행렬 초기화
X_noisy = zeros(num_samples, signal_len); % 입력 (잡음 섞임)
Y_clean = zeros(num_samples, signal_len); % 정답 (깨끗함)
SNR_log = zeros(num_samples, 1);          % SNR 기록용

%% 2. 데이터 생성 루프
fprintf('사인파 데이터 생성 시작... (총 %d개)\n', num_samples);

for i = 1:num_samples
    
    % --- [Step A] 깨끗한 사인파 생성 (I/Q) ---
    % 수식: exp(j * 2 * pi * f * t)
    % 레이더는 I/Q 데이터이므로 sin() 대신 exp()를 씁니다.
    clean_sig = exp(1j * 2 * pi * target_freq * t);
    
    % [중요] 랜덤 위상(Random Phase) 적용
    % 사인파가 항상 0도에서 시작하면 모델이 외워버립니다.
    % 시작 위치를 랜덤하게 돌려줍니다.
    random_phase = exp(1j * 2 * pi * rand); 
    clean_sig = clean_sig * random_phase;
    
    
    % --- [Step B] 복소수 잡음(AWGN) 생성 ---
    % 실수부와 허수부에 각각 가우시안 잡음 추가
    noise = randn(1, signal_len) + 1j * randn(1, signal_len);
    
    
    % --- [Step C] SNR(신호 대 잡음비) 랜덤 설정 ---
    % -5dB(잘 안 보임) ~ 15dB(잘 보임) 사이 랜덤
    target_snr_db = (rand * 20) - 5; 
    
    % 전력 계산 및 잡음 크기 조절
    sig_power = mean(abs(clean_sig).^2);
    noise_power = mean(abs(noise).^2);
    
    desired_noise_power = sig_power / (10^(target_snr_db/10));
    scale_factor = sqrt(desired_noise_power / noise_power);
    
    final_noise = noise * scale_factor;
    noisy_sig = clean_sig + final_noise;
    
    
    % --- [Step D] 정규화 (Normalization) ---
    % 딥러닝 학습을 위해 절댓값 최대치가 1이 되도록 맞춤
    max_val = max(abs(noisy_sig));
    if max_val == 0, max_val = 1; end
    
    noisy_sig = noisy_sig / max_val;
    clean_sig = clean_sig / max_val; 
    
    
    % --- [Step E] 저장 ---
    X_noisy(i, :) = noisy_sig;
    Y_clean(i, :) = clean_sig;
    SNR_log(i) = target_snr_db;
    
    if mod(i, 1000) == 0
        fprintf('%d / %d 완료...\n', i, num_samples);
    end
end

fprintf('데이터 생성 완료!\n');

%% 3. 파일 저장 (필수!)
% -v7.3 옵션은 파이썬 호환성을 위해 꼭 필요합니다.
save('sine_dataset.mat', 'X_noisy', 'Y_clean', 'SNR_log', '-v7.3');
fprintf('sine_dataset.mat 파일로 저장되었습니다.\n');

%% 4. 검증 (툴박스 없이 기본 FFT 사용)
figure;
idx = 1; % 첫 번째 데이터 확인

% [시간 영역] 실수부(Real Part) 비교
subplot(2,1,1);
plot(real(X_noisy(idx,:))); hold on;
plot(real(Y_clean(idx,:)), 'r', 'LineWidth', 2);
title(['Time Domain (Real Part) - SNR: ' num2str(SNR_log(idx)) ' dB']);
legend('Noisy Input', 'Clean Target');
xlabel('Time Index'); ylabel('Amplitude');
grid on;

% [주파수 영역] 기본 FFT 사용 (스펙트로그램 대체)
L = signal_len;
Y = fft(X_noisy(idx,:));
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;

subplot(2,1,2);
plot(f, P1, 'LineWidth', 1.5); 
title('Frequency Domain (FFT)');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 20e6]); % 0 ~ 20MHz 구간만 확대해서 보기
grid on;