# CHES_DPO

### seed 42
SFT HR: 0.03
![HR line](./centered_percentile_pair_experiments/Goodreads/summary_results/plots/line/line_HR.png).
<!-- ![Hdelta_HRe](./centered_percentile_pair_experiments/Goodreads/summary_results/plots/line/delta_line_HR.png). -->

![sequence_logprob_margin_errorbar](./centered_percentile_pair_experiments/Goodreads/summary_plots/sequence_logprob_margin_errorbar.png).

![ln_ches_score](./centered_percentile_pair_experiments/Goodreads/summary_plots/ln_ches_score_errorbar.png).

### seed 0

SFT HR: 0.023
![HR line](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed0/plots/line/line_HR.png).
<!-- ![delta_HR line](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed0/plots/line/delta_line_HR.png). -->

![sequence_logprob_margin_errorbar](./centered_percentile_experiments_4096_1000/Goodreads/centered_percentile_datasets/seed0/summary_plots/sequence_logprob_margin_errorbar.png).

![ln_ches_score](./centered_percentile_experiments_4096_1000/Goodreads/centered_percentile_datasets/seed0/summary_plots/ln_ches_score_errorbar.png).


### seed 1
SFT HR: 0.041
![HR line](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/plots/line/line_HR.png).
<!-- ![delta_HR line](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/plots/line/delta_line_HR.png). -->

![sequence_logprob_margin_errorbar](./centered_percentile_experiments_4096_1000/Goodreads/centered_percentile_datasets/seed1/summary_plots/sequence_logprob_margin_errorbar.png).

![ln_ches_score](./centered_percentile_experiments_4096_1000/Goodreads/centered_percentile_datasets/seed0/summary_plots/ln_ches_score_errorbar.png).




### HIT COUNT BIN
![HitCount_bin](./centered_percentile_pair_experiments/Goodreads/Goodreads_HitCount_bin_lr_1e-6.png).
![HitCount_bin](./centered_percentile_experiments_4096_1000/Goodreads/seed0_Goodreads_HitCount_bin_lr_1e-6.png).
![HitCount_bin](./centered_percentile_experiments_4096_1000/Goodreads/seed1_Goodreads_HitCount_bin_lr_1e-6.png).

### Training State
#### chosen logits  percentile compare seed

##### Using ln_ches score sampling
![ln_ches_score_chosen_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed0/ln_ches_score_chosen_p_compare.png).
##### Using sequence_logprob_margin sampling
![sequence_logprob_margin_chosen_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed0/sequence_logprob_margin_chosen_p_compare.png).

- Min ln ches確實可以避免 chosen下跌

#### reward accuracies percentile compare seed 0
##### Using ln_ches score sampling
![ln_ches_score_accuracies_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed0/ln_ches_score_accuracies_p_compare.png).
##### Using sequence_logprob_margin sampling
![sequence_logprob_margin_accuracies_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed0/sequence_logprob_margin_accuracies_p_compare.png).


#### loss percentile compare seed 0
##### Using ln_ches score sampling
![ln_ches_score_loss_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed0/ln_ches_score_loss_p_compare.png).
##### Using sequence_logprob_margin sampling
![sequence_logprob_margin_loss_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed0/sequence_logprob_margin_loss_p_compare.png).

- 不同的sequence_logprob_margin 影響的是loss下降極限、acc震蕩程度（模型收斂程度）
- 更大的sequence_logprob_margin （easy neg）收縮更快



#### reward accuracies percentile compare seed 1
##### Using ln_ches score sampling
![ln_ches_score_chosen_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/ln_ches_score_chosen_p_compare.png).
##### Using sequence_logprob_margin sampling
![sequence_logprob_margin_chosen_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/sequence_logprob_margin_chosen_p_compare.png).


#### reward accuracies percentile compare seed 1
##### Using ln_ches score sampling
![ln_ches_score_accuracies_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/ln_ches_score_accuracies_p_compare.png).
##### Using sequence_logprob_margin sampling
![sequence_logprob_margin_accuracies_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/sequence_logprob_margin_accuracies_p_compare.png).
### Using avg_token_logprob_margin sampling
![avg_token_logprob_margin_loss_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/avg_token_logprob_margin_accuracies_p_compare.png).


#### loss percentile compare seed 1
##### Using ln_ches score sampling
![ln_ches_score_loss_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/ln_ches_score_loss_p_compare.png).
##### Using sequence_logprob_margin sampling
![sequence_logprob_margin_loss_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/sequence_logprob_margin_loss_p_compare.png).
### Using avg_token_logprob_margin sampling
![avg_token_logprob_margin_loss_p_compare](./centered_percentile_experiments_4096_1000/Goodreads/summary_results/seed1/avg_token_logprob_margin_loss_p_compare.png).

## 另外一個點，目前兩種機率差都仍會受到長度差影響
### seed 0

#### avg_token_logprob_margin_length_bar
![avg_token_logprob_margin_length_bar](./centered_percentile_experiments_4096_1000/Goodreads/centered_percentile_datasets/seed0/summary_plots/avg_token_logprob_margin_length_bar.png).

#### sequence_logprob_margin_length_bar
![sequence_logprob_margin_length_bar](./centered_percentile_experiments_4096_1000/Goodreads/centered_percentile_datasets/seed0/summary_plots/sequence_logprob_margin_length_bar.png).

### seed 1

#### avg_token_logprob_margin_length_bar
![avg_token_logprob_margin_length_bar](./centered_percentile_experiments_4096_1000/Goodreads/centered_percentile_datasets/seed1/summary_plots/avg_token_logprob_margin_length_bar.png).
#### sequence_logprob_margin_length_bar
![sequence_logprob_margin_length_bar](./centered_percentile_experiments_4096_1000/Goodreads/centered_percentile_datasets/seed1/summary_plots/sequence_logprob_margin_length_bar.png).


## 0414
<!-- - Wait to Solve: new Dataset, Origin Dataset num_recommended_unique 差很多問題 （Goodreads lr 1e-6, MovieLens lr 1e-6 new都少一半）
    - 看看shell(是否都是多epoch early stop), 使用的code(evaluate), dataset, eval dataset(sentence bert用一樣的嗎) 
    - code: eval_gini.py無變化（只將id2name.json改成由變數傳入）
    - data: 先從Goodreads檢查起，交叉檢查
        - 用Origin data + new code/shell跑一個it看看
            - 檢查原本SFT code是如何處理train data size > 4096問題
            ```
                train_sample_size = 4096
                train_dataset = load_dataset("json", data_files=train_dataset)
                train_data = train_dataset["train"].shuffle(seed=seed).select(range(train_sample_size))
                val_dataset = load_dataset("json", data_files=valid_dataset)
                val_data = val_dataset["train"].shuffle(seed=seed).select(range(int(train_sample_size/8)))

            ```
            to 
            ```
                train_dataset = load_dataset("json", data_files=train_dataset)
                train_data = train_dataset["train"]
                val_dataset = load_dataset("json", data_files=valid_dataset)
                val_data = val_dataset["train"]
            ```
            還有wandb_projec相關設定刪除
            其餘都一樣，所以直接引用原始SFT model
        - 因為Goodreads/DPO_RN1_new和現在格式一致，所以先測這個
        - 結果：在實驗後，num_recommended_unique數值變回了正常（origin了）
            - 排除code問題
            - [TEST DONE] test data換成我們的，其餘不變
                - [FOUND PROBLEM] test size不一致 Origin: 10645/ NEW: 1000

            - 可進行測試
                - 交換（origin code with new data）:但可能無必要，因為以證明非code問題
                - eval data問題：需查看生成code是否有問題
                - origin data問題：需查看sampled code是否有問題
                    - 雖然非直接檢查，但可以跑看看SFT model的eval
                    - origin data
                    ```
                    [
                    {
                        "model": "./models/SFT_4096/Goodreads/test_result.json",
                        "NDCG": [
                        0.026213088572837034
                        ],
                        "HR": [
                        0.035104186221137604
                        ],
                        "diversity": [
                        1159
                        ],
                        "DivRatio": 0.021783666948595057,
                        "DGU": 0.0591992032127729,
                        "MGU": 0.014938542973758757,
                        "ORRatio": 0.11083544779625976
                    }
                    ]
                    ```
                    new:
                    ```
                    [
                    {
                        "model": "./experiments/predicts/SFT/Goodreads/predicts.json",
                        "NDCG": [
                        0.022619082971107263
                        ],
                        "HR": [
                        0.03
                        ],
                        "diversity": [
                        572
                        ],
                        "DivRatio": 0.1144,
                        "ORRatio": 0.0752
                    }
                    ]
                    ```
                    - 發現SFT就已經有diversity不一致問題了，可能原因限縮到
                        - SFT training data, 也就是sampled data
                            - [checked: no]寫一個code直接檢查是否有很多重複
                            - [checked: 肉眼看無]肉眼（或用code）檢查格式錯誤
                            - [checked by AI]檢查生成code
                        - SFT training code問題已排除，code基本一致
                        - [check: 參數一致]shell傳入參數問題
                        - 用origin data+new code跑一次看看，（理論上基本等價）驗證是否是
                        - [TESTING] 用new evaluate code(inference + eval)跑一次origin的SFT mode: 檢驗是否是SFT訓練時已有錯誤（training data/ training setup）
                            - 就算排除eval code，仍有可能是eval data錯誤
                            - 先檢查code就好，所以test先用原本的

                - training data(generated的)格式、資料問題：需查看生成code是否有問題
                    - 這個可以使用SPRec（origin data）跑一次RN sample generate看看，然後可以不用跑訓練流程，丟比對就好，主要可能錯誤：重複sampled, 格式錯誤
- [solved: 填寫問題] new Dataset, Origin Dataset RN1 HR差很多問題 
- Wait to Solve: MovieLens tail item HR missing
    - 可能原因: 傳入做head/ tail 計算的data不同

- Wait to do: 全部寶跑一次 1 iteration的exp -->

- Wait to Solve: new Dataset, Origin Dataset num_recommended_unique 差很多問題 （Goodreads lr 1e-6, MovieLens lr 1e-6 new都少一半）
    - [FOUND PROBLEM] test size不一致
        - Goodreads Origin: 10645/ NEW: 1000
        - MovieLens Origin: 930128/ NEW: 1000
    - 先查查其他論文用多少test data, 大多都是8:1:1
    - SPRec是先切8:1:1，然後sample 4096, 512, 1000做訓練
- Wait to Solve: CDs_and_Vinyl會有\`"<span class=\"a-size-medium a-color-secondary a-text-normal\"`混入資料問題

- [FOUND_PROBLEM] SPRec是每個it都train一次sft, 然後DPO是在這個第二次SFT上train的
    - 可能要重開一個資料夾跑了
    - SFT部分至少不用重跑
    - ln ches部分不用重跑
    - data部分不用重跑（SPRec, RN1, ln_ches）
    - 他應該是每個it都重新從dataset sample一次，我先用我舊的data跑it0->好像不是喔，都是統一用4096/512那邊sample出來的
        - 要用回SPRec的DPO(沒有early stop的嗎？lr rate)
        - 先跑MovieLens的
    - 跑起來再去寫data_proceccing
    - Goodreads 137152/12566/10654
    - MovieLens 7441019/930127/930128
    - Steam 85956/7412/9482


- [DONE]  Goodreads DPO_RN1 lr = 1e-6 -> check unique_num, DivRatio 是否接近SPRec paper數值
    - DivRatio少paper一半，unique_num少到200多，其餘正常，主要看和ln_ches_scores比較
    - fixed bug，重跑
- [DONE] Goodreads ln_ches_scores lr = 1e-6 -> check 總體有沒有比DPO_RN1好
- [DONE] MovieLens DPO_RN1 lr = 1e-6 -> check tail HR
     <!-- - DivRatio少paper一半，unique_num少到200多，其餘正常，主要看和ln_ches_scores比較 -->
     - HR比paper低，但倒是和之前跑的結果差不多
     - tail HR變0了，可以畫圖看看
- [DONE] origin setup for Goodreads DPO_RN1, ln_ches_scores lr = 0.00002 -> check ln_ches_scores總體有沒有比DPO_RN1好，跟multi epoch early stop 比，決定方法。
    - 舊的是step喔，不知道是不是這個有差
    - 沒有比較好，感覺可以先繼續是multi_epoch調參數，[PENDING]可以畫圖看看
    - 還要跟SPRec paper比，[PENDING]可以考慮跑舊的model/or dataset on test 1000筆，我很懷疑他是否真的是用1000筆
- MovieLens DPO_RN1一下就到acc = 1了，是不是SFT後特別容易擬和，考慮用Max_ln_ches_scores?

- HR很糟，跟paper差一大截，看一下origin(1 epoch)，而且跟SFT model比還下降了
    - 不是test data size的問題，因為之前的用1000 data的HR還是有0.03
    - [PROBLEM_FOUND] sft沒resume
- [IMPO] 舊資料（experiments）可以當without SFT喔（現在看起來跟paper不太一樣，可能是我們是early stop）
# it 0 Goodreads (bug: sft resume from basemodel)

| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| DPO_RN1 | 0.00211 | 0.00398 | 0.00201 | 0.00201 |
| Min_ln_ches_scores | 0.00510 | 0.00797 | 0.00414 | 0.00602 |

| it | DivRatio ↑ | ORRatio ↓ | HR ↑ | NDCG ↑ |
| ----------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| DPO_RN1 | 0.06880 | 0.25300 | 0.00300 | 0.00206 |
| Min_ln_ches_scores | 0.12260 | 0.26480 | 0.00700 | 0.00462 |

| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |
| :--- | ----------: | ----------: | -----------------------: |
| DPO_RN1 | 0.9949 | 0.0262 | 344 |
| Min_ln_ches_scores | 0.9897 | 0.0468 | 613 |

# it 0 Goodreads

| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| DPO_RN1 | 0.04028 | 0.04781 | 0.01145 | 0.01807 |
| Min_ln_ches_scores | 0.04003 | 0.04980 | 0.01204 | 0.02008 |


| it | DivRatio ↑ | ORRatio ↓ | HR ↑ | NDCG ↑ |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| DPO_RN1 | 0.12460 | 0.08620 | 0.03300 | 0.02593 |
| Min_ln_ches_scores | 0.11880 | 0.11300  | 0.03500 | 0.02609 |

| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |
| :--- | ----------: | ----------: | -----------------------: |
| DPO_RN1 | 0.9866 | 0.0475 | 623 |
| Min_ln_ches_scores | 0.9876 | 0.0453 | 594 |

## Goodreads lr 1e-6

| Method | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| DPO_RN1 | 0.04511 | 0.05641 | 0.00842 | 0.01404 |
| SPRec | 0.02961 | 0.03542 | 0.00650 | 0.01112 |
| Min_ln_CHES | **0.04273** | **0.05378** | **0.01094** | **0.01888** |

| it | DivRatio ↑ | ORRatio ↓ | HR ↑ | NDCG ↑ |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| without DPO | 0.1144 | 0.0752 | 0.03 | 0.0226 |
| DPO_RN1 | 0.08355 | 0.09644 | 0.03551 | **0.02701** |
| SPRec | 0.10508 | 0.08952 | 0.02340 | 0.01816 |
| Min_ln_CHES | 0.11616 | 0.09992 | **0.03640** | 0.02690 |


| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |
| :--- | ----------: | ----------: | -----------------------: |
| DPO_RN1 | 0.9864 | 0.0473 | 620.6 |
| SPRec |  0.9849 | 0.0483 | 632.8 |
| Min_ln_CHES | 0.9871 | 0.0443 | 580.8 |
- [PENDING] 畢竟方法都不一樣了，可以試別的lr，先試多點lr的看看結果再試max ln ches
    - [DONE]  Min_ln_ches with lr 1e-5, multi epoch
    - [DONE] terminal 3, gpu 0:DPO_RN1 with lr 1e-5, multi epoch
    - 不行，Min_ln_ches的HR低了，先看看跨方法圖，如無明顯進步，可能可以考慮捨去SFT

- [PENDING] 可以看看acc, 等圖的震盪
- [PENFING] 現在的early stop可能可以改改，改成step看看，先看圖吧
    - 可以用no patience跑五個iteration看看
- [PENDING] base on if chosen is popular or not -> random/min ches
    - [RUNNING] terminal 1, gpu 0:max ln ches
    - [RUNNING] terminal 2, gpu 1:min last hidden scores
    - [RUNNING] terminal 3, gpu 2:min ches

    - 分析max/min ln ches分佈（哪個比較有效壓低ORR峰值，哪些比較會壓低high exposure item等）
    - chosen popular high -> min ches, other random
    - chosen popular low -> min ches, other random
    - chosen popular high -> max ches, other random
    - chosne popular low -> max ches, other random

- [PENDING] 總覺得在RN的基礎上用CHES效果不好
    - 有可能是訓練資料量的差別，直接上4096是是？可以不用再SFT一次
    - SPRec是從輸出機率上直接壓低，如果我也改成用機率？
    - [PENDING]還是用跟CHES原本的方法一樣，先用RN的資料計算CHES SCORE, 然後做persentage的分組測試呢？
        1. 從sampled data 4096筆中再sample出512*5筆資料
        2. 每筆資料從dataset RN pick一個neg sample
        3. 計算每組的 chosen/rejected ln_ches_scores, ches_score, last_hidden_embedding_inner_prods, predict機率差
        4. 用這四個指標分別分五組（所以總共是 ln_ches_scores/0th percentile, ln_ches_scores/25th percentile..., ches_scores/0th percentile, ches_scores/25th percentile,...）總共有4*5個train data
        5. 可以先跑1-3個epoch, lr設0.00002或再高，之後再看要怎麼分析
        6. 可能可以試在raw/few epoch SFT上的差別

## MovieLens 1e-5

| Method | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| DPO_RN1 | 0.00951 | 0.01219 | 0.00000 | 0.00000 |
| SPRec |  0.01019 | 0.01280 | 0.00000 | 0.00000 |
| Min_ln_CHES | 0.00845 | 0.01050 | **0.00401** | **0.00510** |



| it | DivRatio ↑ | ORRatio ↓ | HR ↑ | NDCG ↑ |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| without DPO | 0.127 | 0.1294 | 0.009 | 0.0079 |
| DPO_RN1 | 0.14436 | 0.07896  | 0.00840 | 0.00656 |
| SPRec |0.18168 | 0.04392 | **0.00880** | 0.00701 |
| Min_ln_CHES | 0.13388 | 0.08792  | **0.00880** | **0.00705** |

- head acc != HR ?


| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |
| :--- | ----------: | ----------: | -----------------------: |
| DPO_RN1 | 0.9835 | 0.0551 | 721.8 |
| SPRec | 0.9744 | 0.0693 | 908.4 |
| Min_ln_CHES | 0.9848 | 0.0511 | 669.4 |


## MovieLens 1e-6

| Method | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| DPO_RN1 | 0.01271 | 0.01441 | 0.00000 | 0.00000 |
| SPRec | 0.01262 | 0.01527 | 0.00000 | 0.00000 |
| Min_ln_CHES | 0.01385 | 0.01600 | 0.00000 | 0.00000 |





| it | DivRatio ↑ | ORRatio ↓ | HR ↑ | NDCG ↑ |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| without DPO | 0.127 | 0.1294 | 0.009 | 0.0079 |
| DPO_RN1 | 0.12872 | 0.12580 | 0.01000 | 0.00882 |
| SPRec | 0.15060 | 0.06288  | 0.01060 | 0.00876 |
| Min_ln_CHES |  0.12836 | 0.12568  | 0.01100 | 0.00952 |





| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |
| :--- | ----------: | ----------: | -----------------------: |
| DPO_RN1 | 0.9876 | 0.0491 | 643.6 |
| SPRec | *0.9826 | 0.0574 | 753.0 |
| Min_ln_CHES | 0.9875 | 0.0490 | 641.8 |

# 


# Origin dataset

<!-- ## re-run
| SampleMethod                         | MGU@5 ↓                  | DGU@5 ↓                  | DivRatio@5 ↑            | ORRatio@5 ↓             | NDCG@5 ↑                 | HR@5 ↑                  |
| ------------------------------------ | ------------------------ | ------------------------ | ----------------------- | ----------------------- | ------------------------ | ----------------------- |
| ClusterExposure-DPO (clusterout_low) | <mark>**0.01881**</mark> | <mark>**0.06817**</mark> | <mark>**0.1394**</mark> | 0.102                   | <mark>**0.02011**</mark> | <mark>**0.0310**</mark> |
| SPRec (Baseline)                     | 0.02335                  | 0.08265                  | 0.1256                  | <mark>**0.0906**</mark> | 0.01374                  | 0.0220                  | -->


## Goodreads 1e-6

### Head/Tail @5

| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.04192** | **0.05323** | **0.00788** | **0.01315** |
| **Min_ln_ches** | **0.04397** | **0.05549** | ==**0.00891**== | ==**0.01504**== |


### Overall metrics

| it | DivRatio ↑ | ORRatio ↓ | HR ↑ | NDCG ↑ |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.02329** | **0.12947** | **0.03405** | **0.02564** |
| **Min_ln_ches** | ==**0.02641**== | ==**0.12669**== | ==**0.03614**== | ==**0.02720**== |


| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |
| :--- | ----------: | ----------: | -----------------------: |
| **DPO_RN1** | **0.9491** | **0.3070** | **1245.8** |
| **Min_ln_ches** | ==**0.9391**== | ==**0.3462**== | ==**1405.0**== |


## Steam 1e-6

### Head/Tail @5

| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.01451** | **0.01986** | **0.00740** | **0.01230** |
| **Min_ln_ches** | **0.01431** | **0.01973** | ==**0.00754**== | ==**0.01249**== |



### Overall metrics

| it | DivRatio ↑ | ORRatio ↓ |  HR ↑ | NDCG ↑ |
| ----------- | ----------: |  ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.07340** | **0.04667** | ==**0.01732**== | ==**0.01212**== |
| **Min_ln_ches** | ==**0.07711**== | ==**0.03885**== |  **0.01730** | **0.01203** |

| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |
| :--- | ----------: | ----------: | -----------------------: |
| **DPO_RN1** | **0.9055** | **0.3755** | **3479.4** |
| **Min_ln_ches** | **0.8971** | ==**0.3945**== | ==**3655.2**== |


<!-- ## Steam 1e-5 one epoch

### Head/Tail @5

| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.01369** | **0.01855** | **0.00790** | **0.01230** |
| **Min_ln_ches** | ==**0.01358**== | ==**0.01871**== | ==**0.00834**== | ==**0.01350**== |




### Overall metrics

| it | DivRatio ↑ | ORRatio ↓ |  HR ↑ | NDCG ↑ |
| ----------- | ----------: |  ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.08415** | **0.02695** | **0.01645** | **0.01174** |
| **Min_ln_ches** | **0.08335** | **0.03631** | ==**0.01696**== | ==**0.01182**== | -->

<!-- ## Steam 2e-5

### Head/Tail @5

| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.01439** | **0.01932** | **0.00735** | **0.01230** |
| **Min_ln_ches** | **0.01271** | **0.01741** | ==**0.00794**== | ==**0.01299**== |




### Overall metrics

| it | DivRatio ↑ | ORRatio ↓ | HR ↑ | NDCG ↑ |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.08240** | **0.03831** | ==**0.01696**== | ==**0.01203**== |
| **Min_ln_ches** | **0.08546** | **0.03522** |  **0.01592** | **0.01110** | -->

## Movie_Lens 1e-6 

### Overall metrics

| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.00843** | **0.01106** | **0.00549** | **0.00736** |
| **Min_ln_ches** | ==**0.01125**== | ==**0.01487**== | **0.00389** | **0.00534** |


| it | DivRatio ↑ | ORRatio ↓ | HR ↑ | NDCG ↑ |
| ----------- | ----------: |  ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.03022** | **0.15484** | **0.01364** | **0.01035** |
| **Min_ln_ches** | ==**0.03057**== | **0.16080** |  ==**0.01380**== | ==**0.01038**== |


| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |
| :--- | ----------: | ----------: | -----------------------: |
| **DPO_RN1** | **0.9856** | **0.1415** | **1510.8** |
| **Min_ln_ches** | **0.9859** | ==**0.1432**== | ==**1528.6**== |


<!-- ## CDs_and_Vinly 1e-6  ==試別的 lr(for HR)==

### Head/Tail @5

| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.00729** | **0.01095** | **0.00185** | **0.00322** |
| **Min_ln_ches** | **0.00684** | **0.00865** | ==**0.00196**== | ==**0.00327**== |


### Overall metrics

| it | DivRatio ↑ | ORRatio ↓ |  HR ↑ | NDCG ↑ |
| ----------- | ----------: |  ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.24907** | **0.08352** | ==**0.00522**== | ==**0.00326**== |
| **Min_ln_ches** | ==**0.27390**== | ==**0.06484**== | **0.00466** | **0.00323** | -->


## CD 5e-6
| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |
| ----------- | ----------: | ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.00704** | **0.00980** | **0.00152** | **0.00282** |
| **Min_ln_ches** | **0.00571** | **0.00720** | ==**0.00268**== | ==**0.00453**== |



| it | DivRatio ↑ | ORRatio ↓ |  HR ↑ | NDCG ↑ |
| ----------- | ----------: |  ----------: | ----------: | ----------: |
| **DPO_RN1** | **0.37295** | **0.02022** |**0.00462** | **0.00295** |
| **Min_ln_ches** | **0.32763** | **0.03173** | ==**0.00522**== | ==**0.00347**== |


| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |
| :--- | ----------: | ----------: | -----------------------: |
| **DPO_RN1** | **0.9138** | **0.1841** | **2209.8** |
| **Min_ln_ches** | **0.9286** | **0.1627** | **1952.2** |


- check origin/new codes diff

- MovieLens lr -> 1e-4

- train: origin_train[:-2], origin_valid[:-2], origin_test[:-3]
- valid: origin_train[-1], origin_valid[-1], origin_test[-2]
- test: origin_test[-1]

- base on if chosen is popular or not -> random/min ches
    - history popular
    - chosen popular
