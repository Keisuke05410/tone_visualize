# WAV音楽ファイル波形・EQ解析ツール使用方法

WAV形式の音楽ファイルの波形とEQ特性を視覚化するPythonツールです。

## インストール

```bash
pip install -r requirements_viz.txt
```

## 基本的な使用方法

### 1. 波形表示（基本）
```bash
python visualize_wav.py input/your_audio.wav
```

### 2. 画像として保存
```bash
python visualize_wav.py input/your_audio.wav -o output.png
```

### 3. 特定の時間範囲を表示
```bash
# 10秒から30秒間を表示
python visualize_wav.py input/your_audio.wav -s 10 -d 30
```

## EQ特性解析

### 1. 全体のEQ特性（時間平均）
```bash
python visualize_wav.py input/your_audio.wav --eq
```
- 音楽ファイル全体の平均的な周波数特性を表示
- 線形スケールと対数スケールの両方で表示
- 7つの主要周波数帯域を色分けして表示

### 2. 特定時間点のEQ特性（瞬間解析）
```bash
# mm:ss形式で指定
python visualize_wav.py input/your_audio.wav --instant "1:30"
python visualize_wav.py input/your_audio.wav --instant "2:45"

# 秒数で指定
python visualize_wav.py input/your_audio.wav --instant "90"
```

#### 解析ウィンドウサイズの調整
```bash
# 短いウィンドウ（0.05秒）- より瞬間的な特性
python visualize_wav.py input/your_audio.wav --instant "1:30" --instant-window 0.05

# 長いウィンドウ（0.5秒）- より安定した特性
python visualize_wav.py input/your_audio.wav --instant "1:30" --instant-window 0.5
```

### 3. 時間軸でのEQ変化

#### ヒートマップ表示
```bash
python visualize_wav.py input/your_audio.wav --time-eq
```
- 時間軸でのEQ変化を2Dヒートマップで表示
- 各周波数帯域の時間変化がわかる

#### 3D表示
```bash
python visualize_wav.py input/your_audio.wav --time-3d
```
- 時間-周波数-レベルを3D空間で表示

#### 解析ウィンドウサイズの調整
```bash
# 細かい時間変化を見る（0.5秒ウィンドウ）
python visualize_wav.py input/your_audio.wav --time-eq --window-size 0.5

# 大きなトレンドを見る（2秒ウィンドウ）
python visualize_wav.py input/your_audio.wav --time-eq --window-size 2.0
```

## その他の解析機能

### 1. スペクトログラム
```bash
python visualize_wav.py input/your_audio.wav --spectrogram
```

### 2. オクターブバンド解析
```bash
python visualize_wav.py input/your_audio.wav --octave
```
- ISO 266準拠のオクターブバンド解析
- 音響測定で使用される標準的な周波数帯域

### 3. 周波数帯域強調度解析
```bash
python visualize_wav.py input/your_audio.wav --emphasis
```
- 各周波数帯域がどれだけ強調されているかを分析
- 平均からの偏差と標準偏差による正規化を表示
- 強調度ランキングと音楽的解釈を提供

### 4. すべての解析を実行
```bash
python visualize_wav.py input/your_audio.wav --all
```

## 周波数帯域の説明

| 帯域名 | 周波数範囲 | 音楽的特徴 |
|--------|------------|------------|
| 低域の空気感 | 20-100Hz | 低音の広がり、空間の深さ |
| 低域の太さ | 100-140Hz | キックドラムやベースの太さ、迫力 |
| 低域の安定感と重心 | 220-280Hz | 音楽の土台、安定感を与える帯域 |
| 温かみと粘り | 350-450Hz | 楽器の温かみ、音の厚み |
| 中域のコシと粘り | 800-1000Hz | ボーカルや楽器の存在感、力強さ |
| 明るさと抜け感 | 1800-2200Hz | 音の明るさ、前に出る感覚 |
| エッジ感のある抜け | 3500-4500Hz | 楽器のアタック、音の輪郭 |
| 高域の空気感 | 8000Hz以上 | 高音の煌めき、空間の広がり |

## 数値の意味

### Magnitude (dB)
- 各周波数の音量レベル
- 高い値 = その周波数が強く出ている
- 低い値 = その周波数が弱い

### Average Level (dB)
- 各周波数帯域の平均音量
- 音楽制作でのバランス確認に使用

### 一般的な目安
- **-10dB以上**: 非常に強い
- **-20dB〜-10dB**: 強い
- **-30dB〜-20dB**: 中程度
- **-40dB以下**: 弱い

## 実用例

### 1. 楽曲の構成分析
```bash
# サビの部分のEQ特性
python visualize_wav.py input/song.wav --instant "1:15"

# Aメロの部分のEQ特性
python visualize_wav.py input/song.wav --instant "0:30"
```

### 2. ミックスの品質チェック
```bash
# 全体のバランス確認
python visualize_wav.py input/mix.wav --eq

# 時間軸でのバランス変化
python visualize_wav.py input/mix.wav --time-eq
```

### 3. 楽器の特性分析
```bash
# ドラムソロの瞬間
python visualize_wav.py input/song.wav --instant "2:30" --instant-window 0.1

# ベースラインの特性
python visualize_wav.py input/song.wav --instant "1:00"
```

### 4. 周波数帯域強調度分析
```bash
# 楽曲全体の強調度分析
python visualize_wav.py input/song.wav --emphasis

# 特定の時間範囲での強調度分析
python visualize_wav.py input/song.wav --emphasis -s 60 -d 30
```

## 出力ファイル

### 自動命名規則
```bash
# 基本ファイル
python visualize_wav.py input/song.wav -o analysis.png

# 生成されるファイル
analysis.png          # 波形
analysis_eq.png       # EQ特性
analysis_emphasis.png # 強調度解析
analysis_time_eq.png  # 時間軸EQ変化
analysis_instant_1_30.png  # 1:30時点のEQ
```

## トラブルシューティング

### エラー: ファイルが見つからない
```bash
# ファイルパスを確認
ls input/
```

### エラー: 時間が範囲外
```bash
# 音楽ファイルの長さを確認
python visualize_wav.py input/song.wav
# 出力される「長さ: X.XX秒」を確認
```

### メモリ不足
```bash
# 短い時間範囲で解析
python visualize_wav.py input/song.wav --instant "1:30" -d 10
```

## 高度な使用例

### バッチ処理
```bash
# 複数の時間点を一度に解析
for time in "0:30" "1:00" "1:30" "2:00"; do
    python visualize_wav.py input/song.wav --instant "$time" -o "eq_$time.png"
done
```

### 楽曲の構成分析
```bash
# イントロ、Aメロ、サビの比較
python visualize_wav.py input/song.wav --instant "0:15" -o intro_eq.png
python visualize_wav.py input/song.wav --instant "0:45" -o verse_eq.png
python visualize_wav.py input/song.wav --instant "1:15" -o chorus_eq.png
```

### 強調度解析の活用
```bash
# 楽曲全体の特徴を把握
python visualize_wav.py input/song.wav --emphasis

# 各セクションの強調度を比較
python visualize_wav.py input/song.wav --emphasis -s 0 -d 30 -o intro_emphasis.png
python visualize_wav.py input/song.wav --emphasis -s 60 -d 30 -o chorus_emphasis.png
```

## 強調度解析の読み方

### 強調度スコア（Emphasis Score）
- **+3dB以上**: 非常に強く強調されている
- **+1〜3dB**: 強調されている
- **±1dB**: 平均的
- **-1〜-3dB**: 弱い
- **-3dB以下**: 非常に弱い

### 正規化スコア（Normalized Score）
- **+1σ以上**: 統計的に有意に強い
- **0σ**: 平均的
- **-1σ以下**: 統計的に有意に弱い

### 音楽的解釈の例
- **Mid Body強調**: ボーカルや楽器が前に出ている
- **Low Thickness強調**: 低音に迫力がある
- **Brightness強調**: 明るく抜けの良い音
- **Edge強調**: アタック感がある、鋭い音
- **High Air弱**: 高音が不足、こもった印象

このツールを使用することで、音楽制作、音響分析、音楽理論の学習などに役立てることができます。