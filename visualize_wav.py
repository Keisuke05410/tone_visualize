#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAV音楽ファイルの瞬間EQ解析ツール
特定の時間点でのEQ特性と音楽的解釈を提供します
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


def main():
    parser = argparse.ArgumentParser(description='WAV audio file instant EQ analysis tool')
    parser.add_argument('file', help='WAV file path')
    parser.add_argument('-o', '--output', help='Output image path')
    parser.add_argument('-t', '--time', type=str, help='Time point for analysis (e.g., "1:30" or "90"). If not specified, analyzes the middle of the file.')
    parser.add_argument('-w', '--window', type=float, default=0.1, help='Window size for analysis (seconds, default: 0.1)')
    
    args = parser.parse_args()
    
    # ファイルの存在確認
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return
    
    # 時間点の決定
    if args.time:
        try:
            time_point = parse_time_string(args.time)
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        # 時間が指定されていない場合は、ファイルの中間点を使用
        y, sr = librosa.load(args.file, sr=None)
        total_duration = len(y) / sr
        time_point = total_duration / 2
        print(f"Time point not specified. Using middle of file: {time_point:.2f}s = {int(time_point//60):02d}:{int(time_point%60):02d}")
    
    # 出力パスの準備
    output_path = None
    if args.output:
        if args.time:
            time_str = args.time.replace(':', '_')
        else:
            time_str = f"{time_point:.2f}".replace('.', '_')
        output_path = Path(args.output)
        output_path = output_path.parent / f"{output_path.stem}_instant_{time_str}{output_path.suffix}"
    
    # 瞬間EQ解析を実行
    try:
        plot_instant_eq(args.file, time_point, output_path, args.window)
    except ValueError as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()