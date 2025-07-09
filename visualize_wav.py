#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAV音楽ファイルの波形を視覚化するスクリプト
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import argparse
import os
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_waveform(file_path, output_path=None, duration=None, start_time=0):
    """
    WAVファイルの波形を描画する
    
    Args:
        file_path (str): WAVファイルのパス
        output_path (str): 出力画像のパス（Noneの場合は表示のみ）
        duration (float): 表示する時間の長さ（秒）
        start_time (float): 開始時間（秒）
    """
    
    # 音声ファイルを読み込み
    y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
    
    # 時間軸を作成
    time = np.linspace(0, len(y) / sr, len(y))
    
    # プロットの設定
    plt.figure(figsize=(15, 6))
    
    # ステレオの場合は2チャンネルを別々に表示
    if len(y.shape) > 1:
        plt.subplot(2, 1, 1)
        plt.plot(time, y[:, 0], color='blue', linewidth=0.5)
        plt.title(f'Left Channel - {Path(file_path).name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(time, y[:, 1], color='red', linewidth=0.5)
        plt.title('Right Channel')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
    else:
        # モノラルの場合
        plt.plot(time, y, color='blue', linewidth=0.5)
        plt.title(f'Waveform - {Path(file_path).name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"波形画像を保存しました: {output_path}")
    else:
        plt.show()
    
    return y, sr

def plot_spectrogram(file_path, output_path=None, duration=None, start_time=0):
    """
    WAVファイルのスペクトログラムを描画する
    
    Args:
        file_path (str): WAVファイルのパス
        output_path (str): 出力画像のパス（Noneの場合は表示のみ）
        duration (float): 表示する時間の長さ（秒）
        start_time (float): 開始時間（秒）
    """
    
    # 音声ファイルを読み込み
    y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
    
    # スペクトログラムを計算
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # プロットの設定
    plt.figure(figsize=(15, 6))
    
    # スペクトログラムを表示
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - {Path(file_path).name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"スペクトログラム画像を保存しました: {output_path}")
    else:
        plt.show()

def plot_frequency_spectrum(file_path, output_path=None, duration=None, start_time=0):
    """
    WAVファイルの周波数スペクトラム（EQ特性）を描画する
    
    Args:
        file_path (str): WAVファイルのパス
        output_path (str): 出力画像のパス（Noneの場合は表示のみ）
        duration (float): 表示する時間の長さ（秒）
        start_time (float): 開始時間（秒）
    """
    
    # 音声ファイルを読み込み
    y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
    
    # FFTを計算
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(fft), 1/sr)
    
    # 正の周波数のみを取得
    positive_freq_idx = frequency > 0
    frequency = frequency[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    
    # dBに変換
    magnitude_db = 20 * np.log10(magnitude + 1e-10)  # 小さい値を追加してlog(0)を避ける
    
    # プロットの設定
    plt.figure(figsize=(15, 8))
    
    # 周波数スペクトラムを表示
    plt.subplot(2, 1, 1)
    plt.plot(frequency, magnitude_db, linewidth=0.8)
    plt.title(f'Frequency Spectrum (EQ) - {Path(file_path).name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, sr/2)  # ナイキスト周波数まで
    
    # 対数スケールでも表示
    plt.subplot(2, 1, 2)
    plt.semilogx(frequency, magnitude_db, linewidth=0.8)
    plt.title('Frequency Spectrum (EQ) - Log Scale')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, alpha=0.3)
    plt.xlim(20, sr/2)  # 20Hz以上を表示
    
    # 音楽的に意味のある周波数帯域を表示
    freq_bands = [
        (20, 100, 'Low Air'),
        (100, 140, 'Low Thickness'),
        (220, 280, 'Low Stability'),
        (350, 450, 'Warmth'),
        (800, 1000, 'Mid Body'),
        (1800, 2200, 'Brightness'),
        (3500, 4500, 'Edge'),
        (8000, 20000, 'High Air')
    ]
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink']
    for i, (low, high, name) in enumerate(freq_bands):
        if high <= sr/2:
            plt.axvspan(low, high, alpha=0.1, color=colors[i], label=f'{name} ({low}-{high}Hz)')
    
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"周波数スペクトラム画像を保存しました: {output_path}")
    else:
        plt.show()
    
    # 周波数帯域の解析結果を表示
    print(f"\n周波数帯域解析:")
    for low, high, name in freq_bands:
        if high <= sr/2:
            band_mask = (frequency >= low) & (frequency <= high)
            if np.any(band_mask):
                avg_magnitude = np.mean(magnitude_db[band_mask])
                max_magnitude = np.max(magnitude_db[band_mask])
                print(f"  {name:12} ({low:5}-{high:5}Hz): 平均 {avg_magnitude:6.1f}dB, 最大 {max_magnitude:6.1f}dB")

def plot_octave_analysis(file_path, output_path=None, duration=None, start_time=0):
    """
    WAVファイルのオクターブ解析を行う
    
    Args:
        file_path (str): WAVファイルのパス
        output_path (str): 出力画像のパス（Noneの場合は表示のみ）
        duration (float): 表示する時間の長さ（秒）
        start_time (float): 開始時間（秒）
    """
    
    # 音声ファイルを読み込み
    y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
    
    # オクターブバンドの中心周波数（ISO 266準拠）
    octave_bands = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    octave_levels = []
    
    # FFTを計算
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(fft), 1/sr)
    
    # 正の周波数のみを取得
    positive_freq_idx = frequency > 0
    frequency = frequency[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    
    # 各オクターブバンドのレベルを計算
    for center_freq in octave_bands:
        if center_freq <= sr/2:
            # オクターブバンドの下限と上限を計算
            lower_freq = center_freq / np.sqrt(2)
            upper_freq = center_freq * np.sqrt(2)
            
            # 該当する周波数範囲のマグニチュードを取得
            band_mask = (frequency >= lower_freq) & (frequency <= upper_freq)
            if np.any(band_mask):
                band_magnitude = np.mean(magnitude[band_mask])
                octave_levels.append(20 * np.log10(band_magnitude + 1e-10))
            else:
                octave_levels.append(-100)  # 無音
    
    # プロットの設定
    plt.figure(figsize=(12, 6))
    
    # オクターブバンド解析を表示
    valid_bands = octave_bands[:len(octave_levels)]
    plt.bar(range(len(valid_bands)), octave_levels, color='skyblue', alpha=0.7)
    plt.title(f'Octave Band Analysis - {Path(file_path).name}')
    plt.xlabel('Octave Band Center Frequency (Hz)')
    plt.ylabel('Level (dB)')
    plt.xticks(range(len(valid_bands)), [f'{int(f)}' for f in valid_bands])
    plt.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for i, level in enumerate(octave_levels):
        plt.text(i, level + 1, f'{level:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"オクターブ解析画像を保存しました: {output_path}")
    else:
        plt.show()
    
    # 結果を表示
    print(f"\nオクターブバンド解析結果:")
    for freq, level in zip(valid_bands, octave_levels):
        print(f"  {freq:6.1f}Hz: {level:6.1f}dB")

def plot_time_frequency_analysis(file_path, output_path=None, duration=None, start_time=0, window_size=1.0):
    """
    時間軸でのEQ特性変化を3Dプロットで表示する
    
    Args:
        file_path (str): WAVファイルのパス
        output_path (str): 出力画像のパス（Noneの場合は表示のみ）
        duration (float): 表示する時間の長さ（秒）
        start_time (float): 開始時間（秒）
        window_size (float): 解析ウィンドウサイズ（秒）
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # 音声ファイルを読み込み
    y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
    
    # ウィンドウサイズをサンプル数に変換
    window_samples = int(window_size * sr)
    hop_samples = window_samples // 2  # 50%オーバーラップ
    
    # 時間軸の配列を準備
    time_windows = []
    frequency_data = []
    
    # 各時間ウィンドウで周波数解析
    for i in range(0, len(y) - window_samples, hop_samples):
        window_data = y[i:i + window_samples]
        
        # FFTを計算
        fft = np.fft.fft(window_data)
        magnitude = np.abs(fft)
        frequency = np.fft.fftfreq(len(fft), 1/sr)
        
        # 正の周波数のみを取得
        positive_freq_idx = frequency > 0
        frequency_pos = frequency[positive_freq_idx]
        magnitude_pos = magnitude[positive_freq_idx]
        
        # dBに変換
        magnitude_db = 20 * np.log10(magnitude_pos + 1e-10)
        
        # 時間とデータを保存
        time_windows.append(start_time + i / sr)
        frequency_data.append((frequency_pos, magnitude_db))
    
    # 3Dプロット用のデータを準備
    max_freq = min(8000, sr/2)  # 表示する最大周波数
    freq_bins = np.logspace(np.log10(20), np.log10(max_freq), 100)  # 対数スケール
    
    time_grid = []
    freq_grid = []
    magnitude_grid = []
    
    for t_idx, (time_val, (freq_array, mag_array)) in enumerate(zip(time_windows, frequency_data)):
        # 周波数ビンに対してマグニチュードを補間
        for freq_bin in freq_bins:
            # 最も近い周波数を見つける
            closest_idx = np.argmin(np.abs(freq_array - freq_bin))
            if freq_array[closest_idx] <= max_freq:
                time_grid.append(time_val)
                freq_grid.append(freq_bin)
                magnitude_grid.append(mag_array[closest_idx])
    
    # 3Dプロットを作成
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 散布図として表示
    scatter = ax.scatter(time_grid, freq_grid, magnitude_grid, 
                        c=magnitude_grid, cmap='viridis', alpha=0.6, s=2)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_yscale('log')
    ax.set_title(f'Time-Frequency Analysis - {Path(file_path).name}')
    
    # カラーバーを追加
    plt.colorbar(scatter, ax=ax, label='Magnitude (dB)')
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"時間-周波数解析画像を保存しました: {output_path}")
    else:
        plt.show()

def plot_eq_evolution(file_path, output_path=None, duration=None, start_time=0, window_size=1.0):
    """
    時間軸でのEQ特性変化を2Dヒートマップで表示する
    
    Args:
        file_path (str): WAVファイルのパス
        output_path (str): 出力画像のパス（Noneの場合は表示のみ）
        duration (float): 表示する時間の長さ（秒）
        start_time (float): 開始時間（秒）
        window_size (float): 解析ウィンドウサイズ（秒）
    """
    
    # 音声ファイルを読み込み
    y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
    
    # ウィンドウサイズをサンプル数に変換
    window_samples = int(window_size * sr)
    hop_samples = window_samples // 4  # 25%オーバーラップ
    
    # 音楽的に意味のある周波数帯域を定義
    freq_bands = [
        (20, 100, 'Low Air'),
        (100, 140, 'Low Thickness'),
        (220, 280, 'Low Stability'),
        (350, 450, 'Warmth'),
        (800, 1000, 'Mid Body'),
        (1800, 2200, 'Brightness'),
        (3500, 4500, 'Edge'),
        (8000, 20000, 'High Air')
    ]
    
    # 時間軸とEQデータを準備
    time_windows = []
    eq_evolution = []
    
    # 各時間ウィンドウで周波数解析
    for i in range(0, len(y) - window_samples, hop_samples):
        window_data = y[i:i + window_samples]
        
        # FFTを計算
        fft = np.fft.fft(window_data)
        magnitude = np.abs(fft)
        frequency = np.fft.fftfreq(len(fft), 1/sr)
        
        # 正の周波数のみを取得
        positive_freq_idx = frequency > 0
        frequency_pos = frequency[positive_freq_idx]
        magnitude_pos = magnitude[positive_freq_idx]
        
        # dBに変換
        magnitude_db = 20 * np.log10(magnitude_pos + 1e-10)
        
        # 各周波数帯域の平均レベルを計算
        band_levels = []
        for low, high, name in freq_bands:
            if high <= sr/2:
                band_mask = (frequency_pos >= low) & (frequency_pos <= high)
                if np.any(band_mask):
                    avg_level = np.mean(magnitude_db[band_mask])
                    band_levels.append(avg_level)
                else:
                    band_levels.append(-100)  # 無音
            else:
                band_levels.append(-100)  # 帯域外
        
        time_windows.append(start_time + i / sr)
        eq_evolution.append(band_levels)
    
    # ヒートマップを作成
    eq_evolution = np.array(eq_evolution).T  # 転置して周波数帯域を行にする
    
    plt.figure(figsize=(15, 8))
    
    # ヒートマップを表示
    im = plt.imshow(eq_evolution, aspect='auto', cmap='viridis', 
                   extent=[time_windows[0], time_windows[-1], 0, len(freq_bands)])
    
    # 軸の設定
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency Band')
    plt.title(f'EQ Evolution - {Path(file_path).name}')
    
    # Y軸のラベルを設定
    valid_bands = [name for low, high, name in freq_bands if high <= sr/2]
    plt.yticks(range(len(valid_bands)), valid_bands)
    
    # カラーバーを追加
    cbar = plt.colorbar(im)
    cbar.set_label('Magnitude (dB)')
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"EQ変化ヒートマップを保存しました: {output_path}")
    else:
        plt.show()
    
    # 時間軸での変化を数値で表示
    print(f"\n時間軸でのEQ変化分析:")
    print(f"  解析ウィンドウ: {window_size}秒")
    print(f"  解析ウィンドウ数: {len(time_windows)}")
    print(f"  各周波数帯域の変化範囲:")
    
    for i, (low, high, name) in enumerate(freq_bands):
        if high <= sr/2 and i < len(eq_evolution):
            min_val = np.min(eq_evolution[i])
            max_val = np.max(eq_evolution[i])
            mean_val = np.mean(eq_evolution[i])
            std_val = np.std(eq_evolution[i])
            print(f"    {name:12}: 最小 {min_val:6.1f}dB, 最大 {max_val:6.1f}dB, 平均 {mean_val:6.1f}dB, 標準偏差 {std_val:5.1f}dB")

def parse_time_string(time_str):
    """
    mm:ss形式の時間文字列を秒に変換する
    
    Args:
        time_str (str): "mm:ss" 形式の時間文字列
        
    Returns:
        float: 秒数
    """
    try:
        if ':' in time_str:
            minutes, seconds = time_str.split(':')
            return float(minutes) * 60 + float(seconds)
        else:
            return float(time_str)
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Use 'mm:ss' or seconds.")

def plot_instant_eq(file_path, time_point, output_path=None, window_size=0.1):
    """
    特定の時間点での瞬間的なEQ特性を表示する
    
    Args:
        file_path (str): WAVファイルのパス
        time_point (float): 解析する時間点（秒）
        output_path (str): 出力画像のパス（Noneの場合は表示のみ）
        window_size (float): 解析ウィンドウサイズ（秒）
    """
    
    # 音声ファイルを読み込み
    y, sr = librosa.load(file_path, sr=None)
    
    # 時間点の妥当性をチェック
    total_duration = len(y) / sr
    if time_point < 0 or time_point >= total_duration:
        raise ValueError(f"Time {time_point:.2f}s is out of range. Valid range: 0 - {total_duration:.2f}s")
    
    # 解析ウィンドウのサンプル範囲を計算
    window_samples = int(window_size * sr)
    start_sample = int(time_point * sr) - window_samples // 2
    end_sample = start_sample + window_samples
    
    # 範囲を調整
    if start_sample < 0:
        start_sample = 0
        end_sample = window_samples
    elif end_sample > len(y):
        end_sample = len(y)
        start_sample = end_sample - window_samples
    
    # ウィンドウデータを取得
    window_data = y[start_sample:end_sample]
    actual_start_time = start_sample / sr
    actual_end_time = end_sample / sr
    
    # FFTを計算
    fft = np.fft.fft(window_data)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(fft), 1/sr)
    
    # 正の周波数のみを取得
    positive_freq_idx = frequency > 0
    frequency = frequency[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    
    # dBに変換
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # 音楽的に意味のある周波数帯域を定義
    freq_bands = [
        (20, 100, 'Low Air'),
        (100, 140, 'Low Thickness'),
        (220, 280, 'Low Stability'),
        (350, 450, 'Warmth'),
        (800, 1000, 'Mid Body'),
        (1800, 2200, 'Brightness'),
        (3500, 4500, 'Edge'),
        (8000, 20000, 'High Air')
    ]
    
    # 各帯域の平均レベルを計算
    band_levels = []
    band_names = []
    
    for low, high, name in freq_bands:
        if high <= sr/2:
            band_mask = (frequency >= low) & (frequency <= high)
            if np.any(band_mask):
                avg_level = np.mean(magnitude_db[band_mask])
                band_levels.append(avg_level)
                band_names.append(name)
    
    band_levels = np.array(band_levels)
    
    # 強調度の計算
    overall_mean = np.mean(band_levels)
    overall_std = np.std(band_levels)
    emphasis_scores = band_levels - overall_mean  # 平均からの偏差
    normalized_emphasis = (band_levels - overall_mean) / overall_std  # 標準偏差による正規化
    
    # 強調度ランキング
    emphasis_ranking = np.argsort(emphasis_scores)[::-1]  # 降順
    
    # 音楽的解釈の定義
    musical_interpretations = {
        'Low Air': ('低音の広がり', '空間の深さ'),
        'Low Thickness': ('キック・ベースの太さ', '低音の迫力'),
        'Low Stability': ('音楽の土台', '安定感'),
        'Warmth': ('楽器の温かみ', '音の厚み'),
        'Mid Body': ('ボーカル・楽器の存在感', '音の芯'),
        'Brightness': ('音の明るさ', '前に出る感覚'),
        'Edge': ('楽器のアタック', '音の輪郭'),
        'High Air': ('高音の煌めき', '空間の広がり')
    }

    # プロットの設定
    plt.figure(figsize=(15, 12))
    
    # 上段: 周波数スペクトラム（線形スケール）
    plt.subplot(4, 1, 1)
    plt.plot(frequency, magnitude_db, linewidth=0.8, color='blue')
    plt.title(f'Instant EQ at {time_point:.2f}s ({int(time_point//60):02d}:{int(time_point%60):02d}) - {Path(file_path).name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, min(8000, sr/2))
    
    # 2段目: 周波数スペクトラム（対数スケール）
    plt.subplot(4, 1, 2)
    plt.semilogx(frequency, magnitude_db, linewidth=0.8, color='green')
    plt.title(f'Instant EQ (Log Scale) - Window: {actual_start_time:.2f}s to {actual_end_time:.2f}s')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, alpha=0.3)
    plt.xlim(20, min(8000, sr/2))
    
    # 周波数帯域を色分けして表示
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink']
    for i, (low, high, name) in enumerate(freq_bands):
        if high <= sr/2:
            plt.axvspan(low, high, alpha=0.1, color=colors[i], label=f'{name} ({low}-{high}Hz)')
    
    plt.legend(loc='upper right', fontsize=8)
    
    # 3段目: 強調度解析
    plt.subplot(4, 1, 3)
    emphasis_colors = ['red' if score > 0 else 'blue' for score in emphasis_scores]
    bars = plt.bar(range(len(band_names)), emphasis_scores, color=emphasis_colors, alpha=0.7)
    plt.title(f'Emphasis Analysis at {time_point:.2f}s (Mean: {overall_mean:.1f}dB)')
    plt.xlabel('Frequency Band')
    plt.ylabel('Emphasis Score (dB from mean)')
    plt.xticks(range(len(band_names)), band_names, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Average')
    
    # 値をバーの上に表示
    for i, (bar, score) in enumerate(zip(bars, emphasis_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, score + (0.5 if score > 0 else -0.5), 
                f'{score:+.1f}', ha='center', va='bottom' if score > 0 else 'top', fontsize=8)
    
    plt.legend()
    
    # 4段目: 各周波数帯域の絶対レベル
    plt.subplot(4, 1, 4)
    bars = plt.bar(range(len(band_names)), band_levels, color=colors[:len(band_names)], alpha=0.7)
    plt.title(f'Frequency Band Levels at {time_point:.2f}s')
    plt.xlabel('Frequency Band')
    plt.ylabel('Average Level (dB)')
    plt.xticks(range(len(band_names)), band_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for i, (bar, level) in enumerate(zip(bars, band_levels)):
        plt.text(bar.get_x() + bar.get_width()/2, level + 1, f'{level:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Instant EQ analysis image saved: {output_path}")
    else:
        plt.show()
    
    # 詳細な解析結果を表示
    print(f"\n=== INSTANT EQ & EMPHASIS ANALYSIS ===")
    print(f"Time: {time_point:.2f}s = {int(time_point//60):02d}:{int(time_point%60):02d}")
    print(f"File: {Path(file_path).name}")
    print(f"Analysis window: {actual_start_time:.2f}s - {actual_end_time:.2f}s ({window_size:.2f}s)")
    print(f"Overall mean level: {overall_mean:.1f} dB")
    print(f"Overall std deviation: {overall_std:.1f} dB")
    
    print(f"\n--- EMPHASIS RANKING ---")
    for i, idx in enumerate(emphasis_ranking):
        band_name = band_names[idx]
        score = emphasis_scores[idx]
        norm_score = normalized_emphasis[idx]
        level = band_levels[idx]
        
        # 強調度の評価
        if norm_score > 1:
            emphasis_level = "STRONG"
        elif norm_score > 0:
            emphasis_level = "Moderate"
        elif norm_score > -1:
            emphasis_level = "Weak"
        else:
            emphasis_level = "VERY WEAK"
        
        print(f"{i+1:2d}. {band_name:12} | {score:+6.1f}dB | {norm_score:+5.1f}σ | {emphasis_level:10} | {level:6.1f}dB")
    
    print(f"\n--- INSTANT MUSICAL INTERPRETATION ---")
    for i, idx in enumerate(emphasis_ranking[:3]):  # 上位3つの帯域
        band_name = band_names[idx]
        score = emphasis_scores[idx]
        if score > 0:
            interpretation = musical_interpretations.get(band_name, ('Unknown', 'Unknown'))
            print(f"• {band_name} is emphasized (+{score:.1f}dB)")
            print(f"  → {interpretation[0]} - {interpretation[1]}")
    
    # 弱い帯域の指摘
    weak_bands = [i for i in emphasis_ranking[-2:] if emphasis_scores[i] < -2]
    if weak_bands:
        print(f"\n--- WEAK AREAS ---")
        for idx in weak_bands:
            band_name = band_names[idx]
            score = emphasis_scores[idx]
            print(f"• {band_name} is weak ({score:.1f}dB)")
    
    print(f"\n--- FREQUENCY BAND DETAILS ---")
    for i, (low, high, name) in enumerate(freq_bands):
        if high <= sr/2:
            band_mask = (frequency >= low) & (frequency <= high)
            if np.any(band_mask):
                avg_level = np.mean(magnitude_db[band_mask])
                max_level = np.max(magnitude_db[band_mask])
                print(f"  {name:12} ({low:5}-{high:5}Hz): avg {avg_level:6.1f}dB, max {max_level:6.1f}dB")

def plot_emphasis_analysis(file_path, output_path=None, duration=None, start_time=0):
    """
    周波数帯域の強調度解析を行う
    
    Args:
        file_path (str): WAVファイルのパス
        output_path (str): 出力画像のパス（Noneの場合は表示のみ）
        duration (float): 表示する時間の長さ（秒）
        start_time (float): 開始時間（秒）
    """
    
    # 音声ファイルを読み込み
    y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
    
    # FFTを計算
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(fft), 1/sr)
    
    # 正の周波数のみを取得
    positive_freq_idx = frequency > 0
    frequency = frequency[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    
    # dBに変換
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # 音楽的に意味のある周波数帯域を定義
    freq_bands = [
        (20, 100, 'Low Air'),
        (100, 140, 'Low Thickness'),
        (220, 280, 'Low Stability'),
        (350, 450, 'Warmth'),
        (800, 1000, 'Mid Body'),
        (1800, 2200, 'Brightness'),
        (3500, 4500, 'Edge'),
        (8000, 20000, 'High Air')
    ]
    
    # 各帯域の平均レベルを計算
    band_levels = []
    band_names = []
    
    for low, high, name in freq_bands:
        if high <= sr/2:
            band_mask = (frequency >= low) & (frequency <= high)
            if np.any(band_mask):
                avg_level = np.mean(magnitude_db[band_mask])
                band_levels.append(avg_level)
                band_names.append(name)
    
    band_levels = np.array(band_levels)
    
    # 強調度の計算
    overall_mean = np.mean(band_levels)
    overall_std = np.std(band_levels)
    emphasis_scores = band_levels - overall_mean  # 平均からの偏差
    normalized_emphasis = (band_levels - overall_mean) / overall_std  # 標準偏差による正規化
    
    # 強調度ランキング
    emphasis_ranking = np.argsort(emphasis_scores)[::-1]  # 降順
    
    # 音楽的解釈の定義
    musical_interpretations = {
        'Low Air': ('低音の広がり', '空間の深さ'),
        'Low Thickness': ('キック・ベースの太さ', '低音の迫力'),
        'Low Stability': ('音楽の土台', '安定感'),
        'Warmth': ('楽器の温かみ', '音の厚み'),
        'Mid Body': ('ボーカル・楽器の存在感', '音の芯'),
        'Brightness': ('音の明るさ', '前に出る感覚'),
        'Edge': ('楽器のアタック', '音の輪郭'),
        'High Air': ('高音の煌めき', '空間の広がり')
    }
    
    # プロットの設定
    plt.figure(figsize=(15, 10))
    
    # 上段: 強調度バーグラフ
    plt.subplot(2, 2, 1)
    colors = ['red' if score > 0 else 'blue' for score in emphasis_scores]
    bars = plt.bar(range(len(band_names)), emphasis_scores, color=colors, alpha=0.7)
    plt.title(f'Frequency Band Emphasis Analysis - {Path(file_path).name}')
    plt.xlabel('Frequency Band')
    plt.ylabel('Emphasis Score (dB from mean)')
    plt.xticks(range(len(band_names)), band_names, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Average')
    
    # 値をバーの上に表示
    for i, (bar, score) in enumerate(zip(bars, emphasis_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, score + (0.5 if score > 0 else -0.5), 
                f'{score:+.1f}', ha='center', va='bottom' if score > 0 else 'top', fontsize=8)
    
    plt.legend()
    
    # 右上: 正規化された強調度
    plt.subplot(2, 2, 2)
    colors = ['red' if score > 1 else 'orange' if score > 0 else 'blue' for score in normalized_emphasis]
    bars = plt.bar(range(len(band_names)), normalized_emphasis, color=colors, alpha=0.7)
    plt.title('Normalized Emphasis Score (Standard Deviations)')
    plt.xlabel('Frequency Band')
    plt.ylabel('Normalized Score (σ)')
    plt.xticks(range(len(band_names)), band_names, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Strong emphasis')
    plt.axhline(y=-1, color='blue', linestyle='--', alpha=0.5, label='Weak emphasis')
    
    # 値をバーの上に表示
    for i, (bar, score) in enumerate(zip(bars, normalized_emphasis)):
        plt.text(bar.get_x() + bar.get_width()/2, score + (0.1 if score > 0 else -0.1), 
                f'{score:.1f}σ', ha='center', va='bottom' if score > 0 else 'top', fontsize=8)
    
    plt.legend()
    
    # 下段: 強調度ランキング
    plt.subplot(2, 1, 2)
    ranking_scores = [emphasis_scores[i] for i in emphasis_ranking]
    ranking_names = [band_names[i] for i in emphasis_ranking]
    
    colors = ['red' if score > 2 else 'orange' if score > 0 else 'lightblue' if score > -2 else 'blue' 
              for score in ranking_scores]
    
    bars = plt.barh(range(len(ranking_names)), ranking_scores, color=colors, alpha=0.7)
    plt.title('Emphasis Ranking (Most Emphasized → Least Emphasized)')
    plt.xlabel('Emphasis Score (dB from mean)')
    plt.ylabel('Frequency Band')
    plt.yticks(range(len(ranking_names)), ranking_names)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # 値をバーの右に表示
    for i, (bar, score) in enumerate(zip(bars, ranking_scores)):
        plt.text(score + (0.2 if score > 0 else -0.2), bar.get_y() + bar.get_height()/2, 
                f'{score:+.1f}', ha='left' if score > 0 else 'right', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Emphasis analysis image saved: {output_path}")
    else:
        plt.show()
    
    # 詳細な解析結果を表示
    print(f"\n=== FREQUENCY BAND EMPHASIS ANALYSIS ===")
    print(f"File: {Path(file_path).name}")
    print(f"Overall mean level: {overall_mean:.1f} dB")
    print(f"Overall std deviation: {overall_std:.1f} dB")
    
    print(f"\n--- EMPHASIS RANKING ---")
    for i, idx in enumerate(emphasis_ranking):
        band_name = band_names[idx]
        score = emphasis_scores[idx]
        norm_score = normalized_emphasis[idx]
        level = band_levels[idx]
        
        # 強調度の評価
        if norm_score > 1:
            emphasis_level = "STRONG"
        elif norm_score > 0:
            emphasis_level = "Moderate"
        elif norm_score > -1:
            emphasis_level = "Weak"
        else:
            emphasis_level = "VERY WEAK"
        
        print(f"{i+1:2d}. {band_name:12} | {score:+6.1f}dB | {norm_score:+5.1f}σ | {emphasis_level:10} | {level:6.1f}dB")
    
    print(f"\n--- MUSICAL INTERPRETATION ---")
    for i, idx in enumerate(emphasis_ranking[:3]):  # 上位3つの帯域
        band_name = band_names[idx]
        score = emphasis_scores[idx]
        if score > 0:
            interpretation = musical_interpretations.get(band_name, ('Unknown', 'Unknown'))
            print(f"• {band_name} is emphasized (+{score:.1f}dB)")
            print(f"  → {interpretation[0]} - {interpretation[1]}")
    
    # 弱い帯域の指摘
    weak_bands = [i for i in emphasis_ranking[-2:] if emphasis_scores[i] < -2]
    if weak_bands:
        print(f"\n--- WEAK AREAS ---")
        for idx in weak_bands:
            band_name = band_names[idx]
            score = emphasis_scores[idx]
            print(f"• {band_name} is weak ({score:.1f}dB)")

def main():
    parser = argparse.ArgumentParser(description='WAV audio file waveform and EQ analysis visualization')
    parser.add_argument('file', help='WAV file path')
    parser.add_argument('-o', '--output', help='Output image path')
    parser.add_argument('-d', '--duration', type=float, help='Duration to display (seconds)')
    parser.add_argument('-s', '--start', type=float, default=0, help='Start time (seconds)')
    parser.add_argument('--spectrogram', action='store_true', help='Display spectrogram')
    parser.add_argument('--eq', action='store_true', help='Display EQ characteristics (frequency spectrum)')
    parser.add_argument('--octave', action='store_true', help='Display octave band analysis')
    parser.add_argument('--time-eq', action='store_true', help='Display EQ changes over time as heatmap')
    parser.add_argument('--time-3d', action='store_true', help='Display EQ changes over time as 3D plot')
    parser.add_argument('--instant', type=str, help='Display EQ at specific time point (e.g., "1:30" or "90")')
    parser.add_argument('--emphasis', action='store_true', help='Display frequency band emphasis analysis')
    parser.add_argument('--window-size', type=float, default=1.0, help='Window size for time analysis (seconds, default: 1.0)')
    parser.add_argument('--instant-window', type=float, default=0.1, help='Window size for instant EQ analysis (seconds, default: 0.1)')
    parser.add_argument('--all', action='store_true', help='Display all analyses')
    
    args = parser.parse_args()
    
    # ファイルの存在確認
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return
    
    # 瞬間EQ解析の場合は専用処理
    if args.instant:
        try:
            time_point = parse_time_string(args.instant)
            instant_output = None
            if args.output:
                output_path = Path(args.output)
                instant_output = output_path.parent / f"{output_path.stem}_instant_{args.instant.replace(':', '_')}{output_path.suffix}"
            
            plot_instant_eq(args.file, time_point, instant_output, args.instant_window)
            return
        except ValueError as e:
            print(f"Error: {e}")
            return
    
    # 波形を表示
    print(f"Displaying waveform: {args.file}")
    y, sr = plot_waveform(args.file, args.output, args.duration, args.start)
    
    # 音声ファイルの情報を表示
    print(f"\nAudio file information:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(y) / sr:.2f} seconds")
    print(f"  Channels: {'Stereo' if len(y.shape) > 1 else 'Mono'}")
    
    # 各解析の出力パスを準備
    def get_output_path(suffix):
        if args.output:
            output_path = Path(args.output)
            return output_path.parent / f"{output_path.stem}_{suffix}{output_path.suffix}"
        return None
    
    # EQ特性を表示
    if args.eq or args.all:
        eq_output = get_output_path('eq')
        plot_frequency_spectrum(args.file, eq_output, args.duration, args.start)
    
    # 強調度解析を表示
    if args.emphasis or args.all:
        emphasis_output = get_output_path('emphasis')
        plot_emphasis_analysis(args.file, emphasis_output, args.duration, args.start)
    
    # オクターブバンド解析を表示
    if args.octave or args.all:
        octave_output = get_output_path('octave')
        plot_octave_analysis(args.file, octave_output, args.duration, args.start)
    
    # 時間軸でのEQ変化をヒートマップで表示
    if args.time_eq or args.all:
        time_eq_output = get_output_path('time_eq')
        plot_eq_evolution(args.file, time_eq_output, args.duration, args.start, args.window_size)
    
    # 時間軸でのEQ変化を3Dプロットで表示
    if args.time_3d or args.all:
        time_3d_output = get_output_path('time_3d')
        plot_time_frequency_analysis(args.file, time_3d_output, args.duration, args.start, args.window_size)
    
    # スペクトログラムも表示する場合
    if args.spectrogram or args.all:
        spectrogram_output = get_output_path('spectrogram')
        plot_spectrogram(args.file, spectrogram_output, args.duration, args.start)

if __name__ == "__main__":
    main()