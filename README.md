# Disc_Herniation_Practice — 椎間盤突出 MRI 判讀練習

GitHub（private）：<https://github.com/a0988148626-hash/disc-herniation-practice>

## 專案定位

這是一個練習專案，目的是為達文西機器人脊椎手術影像演算法工作做準備。跟同作者既有的
`Spine_Screw_Practice`（VerSe CT 椎體分割＋螺釘規劃）是同一系列的延伸練習，但目標組織
與影像模態都不同：

| | Spine_Screw_Practice | Disc_Herniation_Practice（本專案） |
|---|---|---|
| 影像模態 | CT | MRI（T1/T2 矢狀面） |
| 目標組織 | 椎體（骨骼） | 椎間盤（軟組織） |
| 核心演算法 | HU 閾值骨骼分割 | 分割遮罩幾何特徵＋輕量分類 |
| 判讀目標 | 螺釘路徑規劃 | 椎間盤突出（disc herniation）二元判讀 |

CT 的 HU 閾值分割邏輯完全不適用於 MRI 軟組織，因此方法從頭設計。

## 資料來源與授權

- **資料集**：SPIDER — *Lumbar spine segmentation in MR images: a dataset and a public benchmark*
  （van der Graaf et al., Scientific Data 2024, <https://doi.org/10.1038/s41597-024-03090-w>）
- **下載來源**：Zenodo, <https://zenodo.org/records/10159290>（Version v4，發布於 2023-11-20，
  2026-11-14 下載當天確認為最新版本）
- **授權**：Creative Commons Attribution 4.0 International（CC BY 4.0）— 2026-11-14 直接開啟
  Zenodo 頁面的「Rights / License」區塊核對原文確認，非僅憑間接搜尋結果。
- **內容**：218 位有下背痛病史的病人、447 份矢狀面 T1/T2（及少量 T2_SPACE）MRI 序列，來自 4 家
  醫院；含椎體／椎間盤／脊椎管分割遮罩，以及獨立的放射科分級 CSV。

### 本次實際下載的檔案（經使用者同意，逐一取得明確同意後才下載）

| 檔案 | 大小 | MD5（Zenodo 官方列出） | 本機驗證結果 |
|---|---|---|---|
| `masks.zip` | 58.2 MB | `49861ff25a0108b3b69d0ce808ae51dc` | 一致 ✅ |
| `overview.csv` | 119.9 kB | `f9a808d2cf67844baa32114a1a769fde` | 大小一致 |
| `radiological_gradings.csv` | 34.4 kB | `a5b7a6ad7fb4daa0bfa650b716f026f6` | 大小一致 |
| `images.zip`（原始 MRI 影像，3.7 GB） | — | — | **未下載，見下方說明** |

### 為什麼沒有下載 images.zip（誠實記錄決策過程）

1. 下載前查過磁碟空間：D 槽僅剩 8.3 GB 可用，C 槽剩 14 GB 可用；SPIDER 全部檔案合計 3.8 GB，
   解壓縮後可能更大，D 槽空間太緊繃，因此專案資料夾改放在 **C 槽**
   （`C:\Ryan Pan\Disc_Herniation_Practice`），經使用者同意。
2. 實測 Zenodo 的下載速度只有約 **84 KB/s**（同一時間測試其他一般網站約 5.4 MB/s，確認是
   Zenodo 伺服器端限流，不是本機網路問題），以此速度下載 3.7 GB 的 `images.zip` 需要約
   **12 小時**。
3. 本方法（見下方「方法設計」）只需要**分割遮罩的幾何邊界**，不需要原始 MRI 的灰階強度值，
   所以 `images.zip` 對本次練習並非必要。
4. 另一個並行工作的 Claude session 曾提議改用 Hugging Face 鏡像下載。查證後發現：
   - 官方對應的 `cdoswald/SPIDER` 其實只是一個 `datasets` 載入腳本，程式碼內
     `_URLS["images"]` 直接指向同一個 `zenodo.org/.../images.zip`，並不會變快。
   - 另外找到一個**非官方第三方重新上傳**的 `MedOtter/SPIDER`（含 900 個 `.mha` 檔案），但這
     未經 Zenodo／原作者背書，正確性與完整性需要額外驗證才能信任，因此最終**沒有使用**。
   - 使用者最終決定：維持原決定，不下載 `images.zip`，只用官方 `masks.zip` + CSV 繼續開發。

這個決定如果未來要延伸到「用原始影像強度做驗證」（例如看 T2 訊號強度判斷椎間盤脫水程度），
需要重新評估：排隊等 Zenodo 慢速下載跑完一夜，或是先抽樣驗證 `MedOtter/SPIDER` 這類第三方
重傳來源的 checksum 一致性後再使用。

## 資料結構（以實際下載到的檔案為準，非憑空假設）

### `masks/*.mha`（447 個檔案，SimpleITK 可讀取，int16，標籤編碼如下）

直接讀取檔案內容確認的標籤規則（與 Hugging Face 上 `cdoswald/SPIDER` README 的官方說明一致）：

```
0         = 背景
1-25      = 椎體，由下往上編號（L5=1, L4=2, L3=3, L2=4, L1=5, T12=6, T11=7, ...）
100       = 脊椎管（spinal canal）
101-125   = 部分可見的椎體
201-225   = 椎間盤，由下往上編號（L5/S1=201, L4/L5=202, ...）
```

座標軸方向（ITK/DICOM LPS 慣例，所有檔案的 direction matrix 皆為單位矩陣，已用其中一個檔案的
metadata 核對）：

```
array[z, y, x]
  z: 上(+) <-> 下(-)   （頭尾軸）
  y: 後(+) <-> 前(-)   （前後軸）
  x: 左(+) <-> 右(-)   （左右／矢狀面切片序號）
```

### `overview.csv`（每個影像序列一列，39 個欄位）

欄位包含 `new_file_name`（對應 mask 檔名）、`num_vertebrae`、`num_discs`、`sex`、`subset`
（**training 360 筆／validation 87 筆**，合計 447，隱藏測試集未釋出）、以及一系列 DICOM
掃描參數（`PixelSpacing`、`SliceThickness`、`Manufacturer` 等）。

序列組成：t1 共 196 筆、t2 共 210 筆、t2_SPACE 共 41 筆（218 位病人中，有些缺 t1 或缺 t2）。

### `radiological_gradings.csv`（1520 列，每個椎間盤層級一列）

欄位：`Patient`、`IVD label`（1~9，由下往上）、`Modic`、`UP endplate`、`LOW endplate`、
`Spondylolisthesis`、**`Disc herniation`（本次判讀目標，二元標籤）**、`Disc narrowing`、
`Disc bulging`、`Pfirrman grade`。

資料品質筆記：有 2 列 `IVD label = 0`（病人 107、152），其餘欄位全為 0 且 Pfirrman grade
欄位為奇怪的佔位值，判斷為資料中的異常／佔位列，已在特徵提取時排除。

218 位病人全部都有分級資料（含 training 與 validation 兩個 subset，隱藏測試集的分級未釋出）。
`Disc herniation = 1` 共 72 列／1520 列（**約 4.7%**），65/218 位病人（約 30%）至少有一個
突出的椎間盤層級——陽性比例偏低，是真實下背痛篩檢族群的合理分布，但也意味著分類任務天生
類別極度不平衡。

## 方法設計：幾何特徵 + 輕量分類（不訓練深度學習模型）

本機沒有 GPU 訓練資源，因此刻意選擇一個「站得住腳但不碰深度學習」的路線：直接利用 SPIDER
官方提供的分割遮罩算幾何特徵，而不是從零訓練分割或分類網路。

### 椎體前緣／後緣的簡化判定

分割遮罩沒有把「椎體（body）」跟「後方結構（椎板、椎弓根、棘突）」分開標註，只有單一的
椎體整體標籤。若直接拿椎體遮罩的最後緣（最大 y）當作「後緣」，會被棘突等後方結構嚴重誤導
（後方結構遠比椎體後緣更靠後）。

採用的簡化方法：以**脊椎管（label 100）的 y 方向中心**為界，把同一 x,z 位置上的椎體體素分成
前（椎體 body）與後（椎板/棘突等）兩群，取前群的最大 y 值當作「椎體後緣」的估計值。這是一個
解剖學上合理但粗略的近似，不是精確的椎體皮質邊界分割。

### 每個椎間盤層級抓取的特徵（`scripts/extract_features.py`）

逐一走過椎間盤遮罩（label = 200 + IVD label）所覆蓋的每個 z（頭尾）切片、每個 x（左右）
切片，計算：

- `disc_height_mm`：椎間盤 z 方向跨距（bounding box 高度，已知有離群值問題，見下方侷限）
- `max_bulge_mm` / `mean_bulge_mm`：椎間盤後緣超出「椎體後緣估計線」的最大／平均距離
- `bulge_normalized`：`max_bulge_mm` 除以相鄰椎體前後徑（AP depth），做尺度正規化
- `asymmetry_mm`：左半邊切片 vs. 右半邊切片的最大突出量差異，粗略偵測是否偏側突出
- `canal_encroach_frac`：有多少比例的切片中，椎間盤後緣已經達到或超過脊椎管前緣
  （代表對脊椎管造成壓迫的比例）
- `mean_disc_ap_extent_mm`：椎間盤在每個切片上的前後徑平均值
- `n_xz_slices_with_disc`：有效切片數量（資料品質參考，不作為分類特徵）

序列選擇：每位病人優先用 `t2`，沒有 t2 才退回 `t1`，兩者都沒有才用 `t2_SPACE`
（T2 對椎間盤軟組織對比度較好，是臨床慣用的判讀序列；這個選擇未逐一比較 t1/t2 兩者的
特徵差異，是方法侷限之一）。

### 驗證方式（`scripts/classify.py`）

1. **規則式門檻（baseline）**：`canal_encroach_frac > 0` 或 `bulge_normalized > 1.0` 即判為
   突出。門檻是憑經驗直接設定的「第一次猜測」，沒有另外調參，刻意保留原始的粗糙結果。
2. **Logistic Regression**：標準化後的 8 個幾何特徵，`class_weight="balanced"` 處理類別不
   平衡，用 **GroupKFold（5 折，依病人分組）** 交叉驗證——同一位病人的所有椎間盤層級永遠
   被分在同一折，避免同一人的資料同時出現在訓練與測試集造成的資訊洩漏。

## 驗證結果（誠實記錄，含失敗與侷限）

資料：1511 個椎間盤層級抓到特徵；因 `canal_encroach_frac` 偶發缺值再剔除 7 列，
最終進入分類評估的是 **1504 列**（72 陽性／1432 陰性，陽性率 4.79%）。

### 規則式門檻結果（很差，且已誠實保留）

| 指標 | 數值 |
|---|---|
| Accuracy | 0.316 |
| Sensitivity（敏感度） | 0.667 |
| Specificity（特異度） | 0.298 |
| PPV（陽性預測值） | 0.046 |
| 混淆矩陣 (tn, fp, fn, tp) | (427, 1005, 24, 48) |

門檻條件幾乎把三分之二的椎間盤都判成「突出」，特異度極低（0.298），代表這個規則對大多數
正常椎間盤也會誤判——**這個簡單門檻法本身幾乎沒有臨床篩檢價值**，只是作為「不做任何學習、
純靠解剖學直覺設門檻」的對照組，用來凸顯接下來 logistic regression 有沒有學到東西。

### Logistic Regression 結果（有一定訊號，但遠不到臨床可用）

| 指標 | 數值 |
|---|---|
| Accuracy | 0.624 |
| Sensitivity | 0.625 |
| Specificity | 0.624 |
| PPV | 0.077 |
| AUC | 0.675 |
| 各折 AUC | 0.581 / 0.723 / 0.809 / 0.657 / 0.731 |
| 混淆矩陣 (tn, fp, fn, tp) | (893, 539, 27, 45) |

誠實解讀：
- AUC 0.675 代表這組幾何特徵確實帶有**超過隨機猜測**的判別力，但離「可用於臨床篩檢」的水準
  （通常期待 AUC > 0.85-0.9）還差得遠。
- PPV 只有 7.7%：在類別極度不平衡（陽性率不到 5%）的情況下，即使敏感度/特異度都有六成多，
  陽性預測值仍然很低——代表如果拿這個模型去篩檢，每抓出約 13 個「疑似突出」，只有 1 個是真的
  突出，另外 12 個是假警報。這是不平衡資料分類的常見陷阱，不是程式錯誤。
- 各折 AUC 從 0.581 到 0.809 波動很大，顯示這個小樣本（約 218 位病人分 5 折，每折僅
  40-45 人）對折間差異很敏感，結果不算穩定。

### 特徵係數（僅供參考，不宜過度解讀）

```
bulge_normalized             -1.181
max_bulge_mm                 +1.122
canal_encroach_frac          -0.797
disc_height_mm               +0.599
mean_bulge_mm                +0.594
asymmetry_mm                 +0.244
ap_depth_mm                  -0.230
mean_disc_ap_extent_mm       +0.054
```

`bulge_normalized` 和 `canal_encroach_frac` 的係數符號是**負的**，跟原本預期（突出量越大
越該判陽性）相反。懷疑是 `max_bulge_mm` 與 `bulge_normalized` 高度共線（後者就是前者除以
`ap_depth_mm`），logistic regression 在共線特徵之間把正負號互相抵銷補償——這組係數不能直接
拿來當作「此特徵越大越不像突出」的臨床結論，只能說整組特徵一起用時有判別力，單一係數的方向
不可靠。這是本方法在解釋性上的已知弱點。

### 已知的失敗模式／資料品質問題

- `disc_height_mm` 用 z 方向 bounding box（最高 z 減最低 z）計算，對任何雜訊或不連續的遮罩
  體素都很敏感：實際看到最大值到 **62.7 mm**（病人 161 層級 2），遠超過正常腰椎椎間盤高度
  （約 8-12 mm），代表 bounding box 高度在某些個案會被遠處的雜訊體素拉爆，是方法的已知缺陷，
  沒有另外加例外處理去「修飾」這個結果。
- 7 個椎間盤層級（病人 35 的層級 4-9、病人 64 的層級 7）在選用的影像序列中完全沒有抓到對應
  遮罩體素，可能是該序列的視野（FOV）沒有覆蓋到那個層級，已直接跳過並在特徵提取的執行紀錄
  中列出，沒有隱瞞。
- 椎體 body/後方結構的區分法（以脊椎管中心為界）是粗略近似，在脊椎管消失或變形的切片（例如
  嚴重病變個案）可能失準，導致 `mean_bulge_mm` 甚至出現負值（代表方法把某些切片的參考線
  估計錯了，不是真的「內縮」）。

## 這個練習對真實臨床判讀的參考價值（誠實評估）

**很有限，但方向正確、流程完整。** 具體來說：

- 有參考價值的部分：確認了「用官方分割遮罩算幾何特徵、配合簡單分類器、GroupKFold 驗證」這條
  流程是可以走通的，且 AUC 0.675 證明幾何特徵本身並非全無訊號——這給未來想做更嚴謹版本（例如
  真的訓練分割網路、或用更精細的椎體解剖結構分離）的人一個「下限基準」可以比較。
- 沒有參考價值／不能直接用於臨床的部分：PPV 只有個位數百分比，規則式門檻法幾乎無鑑別力，
  椎體 body 的分離方法是粗略近似而非解剖學精確分割，`disc_height_mm` 有明顯離群值問題，且
  全部驗證都在同一個資料集上做（沒有外部資料集驗證泛化能力）。這些都代表此方法**不應該被
  當作任何形式的診斷輔助**，純粹是學習分割遮罩幾何分析與分類驗證流程的練習。
- 對達文西機器人脊椎手術影像演算法準備工作的意義：比起 `Spine_Screw_Practice`（骨骼 CT 的
  HU 閾值分割），這次練習到的是「軟組織遮罩的幾何量化＋不平衡資料分類評估＋GroupKFold 避免
  資料洩漏」，這些技能跟骨骼分割不重疊，是另一塊需要補的能力版圖。

## 後續可以怎麼延伸

1. 把椎體 body 跟後方結構的區分法做得更精確（例如用遮罩的連通元件分析，或參考文獻中常見的
   椎體四角點偵測法），取代目前「以脊椎管中心為界」的粗略近似。
2. 修正 `disc_height_mm` 的離群值問題，例如改用椎間盤遮罩在中心幾個切片的高度中位數，而不是
   全域 bounding box。
3. 如果之後真的需要原始 MRI 影像（`images.zip`）：排隊讓 Zenodo 慢速下載跑一整晚，或先驗證
   `MedOtter/SPIDER` 這類第三方重傳來源的 checksum 與官方 `masks.zip` 的對應個案完全一致後
   再採用，不要在未驗證的情況下直接信任。
4. 嘗試用更多特徵（例如椎間盤訊號強度——但這需要 `images.zip`）或更合適的模型（例如
   penalized logistic regression 加 L1 做特徵選擇，或簡單的樹模型），評估是否能把 AUC 推高、
   PPV 改善。
5. 用獨立的外部資料集（而非同一個 SPIDER 資料集的交叉驗證折）驗證泛化能力，才能更誠實地評估
   這套方法的實際價值。

## 使用方式

```bash
# 1. 建立虛擬環境（已建好在 .venv/，Python 3.11 + SimpleITK/numpy/pandas/scikit-learn/scipy/matplotlib）
"C:\Program Files\Python311\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install SimpleITK numpy pandas scikit-learn scipy matplotlib

# 2. 從下載的 masks.zip 解壓縮到乾淨的資料夾（data_extracted/，跟下載檔案分開存放）

# 3. 抓取幾何特徵
.venv\Scripts\python.exe -I scripts\extract_features.py data_extracted\masks\masks data_raw_download\overview.csv data_raw_download\radiological_gradings.csv results\disc_features.csv

# 4. 跑分類驗證
.venv\Scripts\python.exe -I scripts\classify.py results\disc_features.csv results\classification_report.txt
```

## 資料夾結構

```
Disc_Herniation_Practice/
├── README.md                    本檔案
├── .gitignore                   排除資料與虛擬環境
├── scripts/
│   ├── inspect_mask.py          檢查單一 .mha 檔案的標籤分布（探索用）
│   ├── extract_features.py      從分割遮罩抓取椎間盤幾何特徵
│   └── classify.py              規則式門檻 + Logistic Regression 驗證
├── data_raw_download/           下載的原始壓縮檔與 CSV（不進版控，未驗證外部資料）
├── data_extracted/               解壓縮後的 .mha 遮罩檔案（不進版控）
├── results/                      特徵 CSV、分類報告（不進版控，可重跑產生）
└── .venv/                        Python 虛擬環境（不進版控）
```
