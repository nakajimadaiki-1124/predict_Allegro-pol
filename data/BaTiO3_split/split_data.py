import random
from ase.io import read, write

# --- 設定 ---

# 入力ファイル名
INPUT_FILE = "BaTiO3.xyz"

# 出力ファイル名
OUTPUT_TEST_FILE = "test_data.xyz"      # 6個のテストデータ用
OUTPUT_TRAIN_VAL_FILE = "train_val_data.xyz" # 53個の学習・検証データ用

# テストデータとして分割する構造の数
NUM_TEST_SAMPLES = 6

# ランダムシード（毎回同じ結果にするために固定）
RANDOM_SEED = 123456


# --- プログラム本体 ---

print(f"'{INPUT_FILE}' を読み込んでいます...")
# index=':' を指定することで、XYZファイル内の全ての構造をリストとして読み込む
try:
    all_structures = read(INPUT_FILE, index=':')
    print(f"合計 {len(all_structures)} 個の構造を読み込みました。")
except FileNotFoundError:
    print(f"エラー: '{INPUT_FILE}' が見つかりません。パスを確認してください。")
    exit()

# ランダムシードを設定
random.seed(RANDOM_SEED)

# 読み込んだ構造のリストをランダムに並び替え（シャッフル）
random.shuffle(all_structures)
print("構造のリストをランダムにシャッフルしました。")

# データを分割
test_structures = all_structures[:NUM_TEST_SAMPLES]
train_val_structures = all_structures[NUM_TEST_SAMPLES:]
print(f"テスト用に {len(test_structures)} 個、学習・検証用に {len(train_val_structures)} 個の構造に分割します。")

# 分割したデータをそれぞれXYZファイルに書き込み
write(OUTPUT_TEST_FILE, test_structures)
print(f"テストデータを '{OUTPUT_TEST_FILE}' に保存しました。")

write(OUTPUT_TRAIN_VAL_FILE, train_val_structures)
print(f"学習・検証データを '{OUTPUT_TRAIN_VAL_FILE}' に保存しました。")

print("\n処理が完了しました。")