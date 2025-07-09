# WAV音楽ファイル瞬間EQ解析ツール使用方法

WAV形式の音楽ファイルの特定時間点でのEQ特性を詳細に解析するPythonツールです。

## インストール

```bash
pip install -r requirements_viz.txt
```

## 基本的な使用方法

### 1. 瞬間EQ解析（基本）
```bash
# ファイルの中間点を自動解析
python visualize_wav.py input/your_audio.wav
```

### 2. 特定時間点のEQ解析
```bash
# mm:ss形式で時間指定
python visualize_wav.py input/your_audio.wav -t "1:30"
python visualize_wav.py input/your_audio.wav -t "2:45"

# 秒数で時間指定
python visualize_wav.py input/your_audio.wav -t "90"
```

### 3. 画像として保存
```bash
python visualize_wav.py input/your_audio.wav -t "1:30" -o output.png
```

## 詳細設定

### 解析ウィンドウサイズの調整
```bash
# 短いウィンドウ（0.05秒）- より瞬間的な特性
python visualize_wav.py input/your_audio.wav -t "1:30" -w 0.05

# 長いウィンドウ（0.5秒）- より安定した特性
python visualize_wav.py input/your_audio.wav -t "1:30" -w 0.5
```

### コマンドライン引数一覧
```bash
python visualize_wav.py [-h] [-o OUTPUT] [-t TIME] [-w WINDOW] file

positional arguments:
  file                 WAV file path

options:
  -h, --help           show this help message and exit
  -o, --output OUTPUT  Output image path
  -t, --time TIME      Time point for analysis (e.g., "1:30" or "90")
  -w, --window WINDOW  Window size for analysis (seconds, default: 0.1)
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
python visualize_wav.py input/song.wav -t "1:15"

# Aメロの部分のEQ特性
python visualize_wav.py input/song.wav -t "0:30"

# 中間点の自動解析
python visualize_wav.py input/song.wav
```

### 2. ミックスの品質チェック
```bash
# 楽曲の重要な部分を解析
python visualize_wav.py input/mix.wav -t "1:00"

# 複数の時間点を比較
python visualize_wav.py input/mix.wav -t "0:30" -o verse.png
python visualize_wav.py input/mix.wav -t "1:15" -o chorus.png
```

### 3. 楽器の特性分析
```bash
# ドラムソロの瞬間
python visualize_wav.py input/song.wav -t "2:30" -w 0.1

# ベースラインの特性
python visualize_wav.py input/song.wav -t "1:00"
```

### 4. 詳細な時間分析
```bash
# 短いウィンドウで瞬間的な特性を見る
python visualize_wav.py input/song.wav -t "1:30" -w 0.05

# 長いウィンドウで安定した特性を見る
python visualize_wav.py input/song.wav -t "1:30" -w 0.5
```

## 出力ファイル

### 自動命名規則
```bash
# 時間指定ありの場合
python visualize_wav.py input/song.wav -t "1:30" -o analysis.png
# 生成されるファイル: analysis_instant_1_30.png

# 時間指定なしの場合（中間点解析）
python visualize_wav.py input/song.wav -o analysis.png
# 生成されるファイル: analysis_instant_45_23.png (45.23秒時点)
```

## トラブルシューティング

### エラー: ファイルが見つからない
```bash
# ファイルパスを確認
ls input/
```

### エラー: 時間が範囲外
```bash
# 音楽ファイルの長さを確認するため、中間点解析を実行
python visualize_wav.py input/song.wav
# 出力される「Time point not specified. Using middle of file: X.XX秒」を確認
```

### メモリ不足
```bash
# より短いウィンドウサイズで解析
python visualize_wav.py input/song.wav -t "1:30" -w 0.05
```

## 高度な使用例

### バッチ処理
```bash
# 複数の時間点を一度に解析
for time in "0:30" "1:00" "1:30" "2:00"; do
    python visualize_wav.py input/song.wav -t "$time" -o "eq_${time//:/_}.png"
done
```

### 楽曲の構成分析
```bash
# イントロ、Aメロ、サビの比較
python visualize_wav.py input/song.wav -t "0:15" -o intro_eq.png
python visualize_wav.py input/song.wav -t "0:45" -o verse_eq.png
python visualize_wav.py input/song.wav -t "1:15" -o chorus_eq.png
```

### 時間軸での詳細分析
```bash
# 10秒間隔で楽曲全体を解析
for i in {0..180..10}; do
    python visualize_wav.py input/song.wav -t "$i" -o "timeline_${i}s.png"
done
```

### 異なるウィンドウサイズでの比較
```bash
# 同じ時間点を異なるウィンドウサイズで解析
python visualize_wav.py input/song.wav -t "1:30" -w 0.05 -o precise_eq.png
python visualize_wav.py input/song.wav -t "1:30" -w 0.1 -o normal_eq.png
python visualize_wav.py input/song.wav -t "1:30" -w 0.5 -o smooth_eq.png
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

## 注意事項

- このツールは瞬間EQ解析に特化しています
- 時間を指定しない場合、ファイルの中間点を自動で解析します
- ウィンドウサイズを小さくすると、より瞬間的な特性が見えますが、ノイズも増加します
- ウィンドウサイズを大きくすると、より安定した特性が見えますが、瞬間的な変化は見えにくくなります