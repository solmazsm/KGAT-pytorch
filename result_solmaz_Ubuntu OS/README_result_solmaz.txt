# Solmaz result

## Reference
# Knowledge Graph Attention Network
This is PyTorch implementation for the paper:
>Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. [Paper in ACM DL](https://dl.acm.org/authorize.cfm?key=N688414) or [Paper in arXiv](https://arxiv.org/abs/1905.07854). In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.
*********************************************************


## Environment Requirements

- Ubuntu OS
- Python >= 3.8 (Anaconda3 is recommended)
- PyTorch 1.7+
- A Nvidia GPU with cuda 11.1+

## Datasets

We user three popular datasets Amazon-book, Last-FM, and Yelp2018 to conduct experiments.
* We follow the paper "[KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854)" to process data.
* In order to construct cold-start scenario, we find user registration time, item publication time or first interaction time in the full version of recommendation datasets. Then we divide new and old ones in chronological order. 
* For Amazon-book, download book reviews (5-core) from [here](http://jmcauley.ucsd.edu/data/amazon), put it into related `rawdata` folder.
* For Last-FM, download LFM-1b dataset from [here](http://www.cp.jku.at/datasets/LFM-1b/), unzip and put it into related `rawdata` folder.
* For Yelp2018, download Yelp2018 version dataset from [here](https://www.heywhale.com/mw/dataset/5ecbc342fac16e0036ec41a0),  unzip and put it into related `rawdata` folder.

The prepared folder structure is like this:

```
- Datasets
    - pretrain
    - amazon-book
	- rawdata
		- reviews_Books_5.json.gz
    - last-fm
    	- rawdata
    		- LFM-1b_albums.txt
    		- LFM-1b_artists.txt
    		- ...
    - yelp2018
    	- rawdata
    		- yelp_academic_dataset_business.json
    		- yelp_academic_dataset_checkin.json
    		- ...
```

## Train

1. Now, we have provided the cold-start scenario data of last-fm. The codes for constructing the other datasets is as follows.
   ```shell
   python construct_data.py
   ```

2. Start training
   # Run the Codes
  
   ```shell
   python3 main_kgat.py 
   python3 main_ecfkg.py
   python3 main_cke.py
   python3 main_bprmf.py
   python3 main_nfm.py  
   ```


## Reference

# Knowledge Graph Attention Network
This is PyTorch implementation for the paper:
>Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. [Paper in ACM DL](https://dl.acm.org/authorize.cfm?key=N688414) or [Paper in arXiv](https://arxiv.org/abs/1905.07854). In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.

You can find Tensorflow implementation by the paper authors [here](https://github.com/xiangwang1223/knowledge_graph_attention_network).

## Introduction
Knowledge Graph Attention Network (KGAT) is a new recommendation framework tailored to knowledge-aware personalized recommendation. Built upon the graph neural network framework, KGAT explicitly models the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.

If you want to use codes and datasets in your research, please contact the paper authors and cite the following paper as the reference:
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat{-}Seng Chua},
  title     = {{KGAT:} Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  pages     = {950--958},
  year      = {2019}
}
```

## Environment Requirement
The code has been tested running under Python 3.7.10. The required packages are as follows:
* torch == 1.6.0
* numpy == 1.21.4
* pandas == 1.3.5
* scipy == 1.5.2
* tqdm == 4.62.3
* scikit-learn == 1.0.1

## Run the Codes
* FM
```
python main_nfm.py --model_type fm --data_name amazon-book
```
* NFM
```
python main_nfm.py --model_type nfm --data_name amazon-book
```
* BPRMF
```
python main_bprmf.py --data_name amazon-book
```
* ECFKG
```
python main_ecfkg.py --data_name amazon-book
```
* CKE
```
python main_cke.py --data_name amazon-book
```
* KGAT
```
python main_kgat.py --data_name amazon-book
```

## Results
With my code, following are the results of each model when training with dataset `amazon-book`.

| Model                                             | Best Epoch | Precision@20 | Recall@20 | NDCG@20 |
| :---:                                             | :---:      | :---:        | :---:     | :---:   |
| FM                                                | 370        | 0.0154       | 0.1478    | 0.0784  |
| NFM                                               | 140        | 0.0137       | 0.1309    | 0.0696  |
| BPRMF                                             | 330        | 0.0146       | 0.1395    | 0.0736  |
| ECFKG                                             |  10        | 0.0134       | 0.1264    | 0.0663  |
| CKE                                               | 320        | 0.0145       | 0.1394    | 0.0733  |
| KGAT <br> (agg: bi-interaction; lap: random-walk) | 280        | 0.0150       | 0.1440    | 0.0766  |
| KGAT <br> (agg: bi-interaction; lap: symmetric)   | 200        | 0.0149       | 0.1428    | 0.0755  |
| KGAT <br> (agg: graphsage;      lap: random-walk) | 450        | 0.0147       | 0.1430    | 0.0747  |
| KGAT <br> (agg: graphsage;      lap: symmetric)   | 160        | 0.0146       | 0.1410    | 0.0735  |
| KGAT <br> (agg: gcn;            lap: random-walk) | 280        | 0.0149       | 0.1440    | 0.0760  |
| KGAT <br> (agg: gcn;            lap: symmetric)   | 670        | 0.0150       | 0.1448    | 0.0768  |

## Related Papers
* FM
    * Proposed in [Fast context-aware recommendations with factorization machines](https://dl.acm.org/citation.cfm?id=2010002), SIGIR 2011.

* NFM
    * Proposed in [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777), SIGIR 2017.

* BPRMF
    * Proposed in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/citation.cfm?id=1795167), UAI 2009.
    * Key point: 
        * Replace point-wise with pair-wise.

* ECFKG
    * Proposed in [Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation](https://arxiv.org/abs/1805.03352), Algorithms 2018.
    * Implementation by the paper authors: [https://github.com/evison/KBE4ExplainableRecommendation](https://github.com/evison/KBE4ExplainableRecommendation)
    * Key point: 
        * Introduce Knowledge Graph to Collaborative Filtering

* CKE
    * Proposed in [Collaborative Knowledge Base Embedding for Recommender Systems](https://dl.acm.org/citation.cfm?id=2939673), KDD 2016.
    * Key point: 
        * Leveraging structural content, textual content and visual content from the knowledge base.
        * Use TransR which is an approach for heterogeneous network, to represent entities and relations in distinct semantic space bridged by relation-specific matrices.
        * Performing knowledge base embedding and collaborative filtering jointly.

* KGAT
    * Proposed in [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854), KDD 2019.
    * Implementation by the paper authors: [https://github.com/xiangwang1223/knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network)
    * Key point:
        * Model the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.
        * Train KG part and CF part in turns.
        
****************************************************
* KGAT model amazon-book

****************************************************
solmaz@VirtualBox:~/sol$ python3 main_kgat.py 
All logs will be saved to trained_model/KGAT/amazon-book/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain1/log3.log
2022-06-09 09:36:16,541 - root - INFO - Namespace(Ks='[20, 40, 60, 80, 100]', aggregation_type='bi-interaction', cf_batch_size=1024, cf_l2loss_lambda=1e-05, cf_print_every=1, conv_dim_list='[64, 32, 16]', data_dir='datasets/', data_name='amazon-book', embed_dim=64, evaluate_every=10, kg_batch_size=2048, kg_l2loss_lambda=1e-05, kg_print_every=1, laplacian_type='random-walk', lr=0.0001, mess_dropout='[0.1, 0.1, 0.1]', n_epoch=1000, pretrain_embedding_dir='datasets/pretrain/', pretrain_model_path='trained_model/model.pth', relation_dim=64, save_dir='trained_model/KGAT/amazon-book/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain1/', seed=2019, stopping_steps=10, test_batch_size=10000, use_pretrain=1)
2022-06-09 09:49:17,787 - root - INFO - n_users:           70679
2022-06-09 09:49:17,793 - root - INFO - n_items:           24915
2022-06-09 09:49:17,793 - root - INFO - n_entities:        113487
2022-06-09 09:49:17,793 - root - INFO - n_users_entities:  184166
2022-06-09 09:49:17,793 - root - INFO - n_relations:       80
2022-06-09 09:49:17,793 - root - INFO - n_h_list:          6420520
2022-06-09 09:49:17,793 - root - INFO - n_t_list:          6420520
2022-06-09 09:49:17,794 - root - INFO - n_r_list:          6420520
2022-06-09 09:49:17,794 - root - INFO - n_cf_train:        652514
2022-06-09 09:49:17,794 - root - INFO - n_cf_test:         193920
2022-06-09 09:49:17,794 - root - INFO - n_kg_train:        6420520
/home/solmaz/MetaKG/data_loader/loader_kgat.py:118: RuntimeWarning: divide by zero encountered in power
  d_inv = np.power(rowsum, -1.0).flatten()
2022-06-09 09:49:43,899 - root - INFO - KGAT(
  (entity_user_embed): Embedding(184166, 64)
  (relation_embed): Embedding(80, 64)
  (aggregator_layers): ModuleList(
    (0): Aggregator(
      (message_dropout): Dropout(p=0.1, inplace=False)
      (activation): LeakyReLU(negative_slope=0.01)
      (linear1): Linear(in_features=64, out_features=64, bias=True)
      (linear2): Linear(in_features=64, out_features=64, bias=True)
    )
    (1): Aggregator(
      (message_dropout): Dropout(p=0.1, inplace=False)
      (activation): LeakyReLU(negative_slope=0.01)
      (linear1): Linear(in_features=64, out_features=32, bias=True)
      (linear2): Linear(in_features=64, out_features=32, bias=True)
    )
    (2): Aggregator(
      (message_dropout): Dropout(p=0.1, inplace=False)
      (activation): LeakyReLU(negative_slope=0.01)
      (linear1): Linear(in_features=32, out_features=16, bias=True)
      (linear2): Linear(in_features=32, out_features=16, bias=True)
    )
  )
)
2022-06-09 09:50:00,131 - root - INFO - CF Training: Epoch 0001 Iter 0001 / 0638 | Time 16.2s | Iter Loss 0.0299 | Iter Mean Loss 0.0299
2022-06-09 09:50:14,876 - root - INFO - CF Training: Epoch 0001 Iter 0002 / 0638 | Time 14.7s | Iter Loss 0.0239 | Iter Mean Loss 0.0269
2022-06-09 09:50:28,632 - root - INFO - CF Training: Epoch 0001 Iter 0003 / 0638 | Time 13.8s | Iter Loss 0.0270 | Iter Mean Loss 0.0269
2022-06-09 09:50:43,003 - root - INFO - CF Training: Epoch 0001 Iter 0004 / 0638 | Time 14.4s | Iter Loss 0.0222 | Iter Mean Loss 0.0257
2022-06-09 09:51:01,137 - root - INFO - CF Training: Epoch 0001 Iter 0005 / 0638 | Time 18.1s | Iter Loss 0.0248 | Iter Mean Loss 0.0256
2022-06-09 09:51:20,184 - root - INFO - CF Training: Epoch 0001 Iter 0006 / 0638 | Time 19.0s | Iter Loss 0.0249 | Iter Mean Loss 0.0255
2022-06-09 09:51:29,194 - root - INFO - CF Training: Epoch 0001 Iter 0007 / 0638 | Time 9.0s | Iter Loss 0.0230 | Iter Mean Loss 0.0251
2022-06-09 09:51:37,133 - root - INFO - CF Training: Epoch 0001 Iter 0008 / 0638 | Time 7.9s | Iter Loss 0.0237 | Iter Mean Loss 0.0249
2022-06-09 09:51:50,847 - root - INFO - CF Training: Epoch 0001 Iter 0009 / 0638 | Time 13.7s | Iter Loss 0.0237 | Iter Mean Loss 0.0248
2022-06-09 09:52:02,510 - root - INFO - CF Training: Epoch 0001 Iter 0010 / 0638 | Time 11.7s | Iter Loss 0.0219 | Iter Mean Loss 0.0245
2022-06-09 09:52:11,004 - root - INFO - CF Training: Epoch 0001 Iter 0011 / 0638 | Time 8.5s | Iter Loss 0.0255 | Iter Mean Loss 0.0246
2022-06-09 09:52:22,377 - root - INFO - CF Training: Epoch 0001 Iter 0012 / 0638 | Time 11.4s | Iter Loss 0.0236 | Iter Mean Loss 0.0245
2022-06-09 09:52:35,057 - root - INFO - CF Training: Epoch 0001 Iter 0013 / 0638 | Time 12.7s | Iter Loss 0.0290 | Iter Mean Loss 0.0249
2022-06-09 09:52:46,915 - root - INFO - CF Training: Epoch 0001 Iter 0014 / 0638 | Time 11.8s | Iter Loss 0.0249 | Iter Mean Loss 0.0249
2022-06-09 09:52:57,157 - root - INFO - CF Training: Epoch 0001 Iter 0015 / 0638 | Time 10.2s | Iter Loss 0.0287 | Iter Mean Loss 0.0251
2022-06-09 09:53:06,477 - root - INFO - CF Training: Epoch 0001 Iter 0016 / 0638 | Time 9.3s | Iter Loss 0.0260 | Iter Mean Loss 0.0252
2022-06-09 09:53:17,662 - root - INFO - CF Training: Epoch 0001 Iter 0017 / 0638 | Time 11.2s | Iter Loss 0.0281 | Iter Mean Loss 0.0254
2022-06-09 09:53:28,857 - root - INFO - CF Training: Epoch 0001 Iter 0018 / 0638 | Time 11.2s | Iter Loss 0.0260 | Iter Mean Loss 0.0254
2022-06-09 09:53:38,190 - root - INFO - CF Training: Epoch 0001 Iter 0019 / 0638 | Time 9.3s | Iter Loss 0.0293 | Iter Mean Loss 0.0256
2022-06-09 09:53:51,122 - root - INFO - CF Training: Epoch 0001 Iter 0020 / 0638 | Time 12.9s | Iter Loss 0.0265 | Iter Mean Loss 0.0256
2022-06-09 09:54:09,045 - root - INFO - CF Training: Epoch 0001 Iter 0021 / 0638 | Time 17.9s | Iter Loss 0.0270 | Iter Mean Loss 0.0257
2022-06-09 09:54:29,016 - root - INFO - CF Training: Epoch 0001 Iter 0022 / 0638 | Time 20.0s | Iter Loss 0.0288 | Iter Mean Loss 0.0258
2022-06-09 09:54:41,753 - root - INFO - CF Training: Epoch 0001 Iter 0023 / 0638 | Time 12.7s | Iter Loss 0.0239 | Iter Mean Loss 0.0258
2022-06-09 09:54:54,104 - root - INFO - CF Training: Epoch 0001 Iter 0024 / 0638 | Time 12.3s | Iter Loss 0.0236 | Iter Mean Loss 0.0257
2022-06-09 09:55:06,776 - root - INFO - CF Training: Epoch 0001 Iter 0025 / 0638 | Time 12.7s | Iter Loss 0.0266 | Iter Mean Loss 0.0257
2022-06-09 09:55:19,961 - root - INFO - CF Training: Epoch 0001 Iter 0026 / 0638 | Time 13.2s | Iter Loss 0.0235 | Iter Mean Loss 0.0256
2022-06-09 09:55:33,742 - root - INFO - CF Training: Epoch 0001 Iter 0027 / 0638 | Time 13.8s | Iter Loss 0.0239 | Iter Mean Loss 0.0256
2022-06-09 09:55:46,537 - root - INFO - CF Training: Epoch 0001 Iter 0028 / 0638 | Time 12.8s | Iter Loss 0.0256 | Iter Mean Loss 0.0256
2022-06-09 09:55:59,756 - root - INFO - CF Training: Epoch 0001 Iter 0029 / 0638 | Time 13.2s | Iter Loss 0.0263 | Iter Mean Loss 0.0256
2022-06-09 09:56:12,606 - root - INFO - CF Training: Epoch 0001 Iter 0030 / 0638 | Time 12.8s | Iter Loss 0.0269 | Iter Mean Loss 0.0256
2022-06-09 09:56:24,925 - root - INFO - CF Training: Epoch 0001 Iter 0031 / 0638 | Time 12.3s | Iter Loss 0.0277 | Iter Mean Loss 0.0257
2022-06-09 09:56:37,620 - root - INFO - CF Training: Epoch 0001 Iter 0032 / 0638 | Time 12.7s | Iter Loss 0.0222 | Iter Mean Loss 0.0256
2022-06-09 09:56:53,095 - root - INFO - CF Training: Epoch 0001 Iter 0033 / 0638 | Time 15.5s | Iter Loss 0.0308 | Iter Mean Loss 0.0257
2022-06-09 09:57:13,850 - root - INFO - CF Training: Epoch 0001 Iter 0034 / 0638 | Time 20.8s | Iter Loss 0.0281 | Iter Mean Loss 0.0258
2022-06-09 09:57:28,700 - root - INFO - CF Training: Epoch 0001 Iter 0035 / 0638 | Time 14.8s | Iter Loss 0.0240 | Iter Mean Loss 0.0258
2022-06-09 09:57:42,330 - root - INFO - CF Training: Epoch 0001 Iter 0036 / 0638 | Time 13.6s | Iter Loss 0.0237 | Iter Mean Loss 0.0257
2022-06-09 09:57:54,329 - root - INFO - CF Training: Epoch 0001 Iter 0037 / 0638 | Time 12.0s | Iter Loss 0.0247 | Iter Mean Loss 0.0257
2022-06-09 09:58:06,806 - root - INFO - CF Training: Epoch 0001 Iter 0038 / 0638 | Time 12.5s | Iter Loss 0.0271 | Iter Mean Loss 0.0257
2022-06-09 09:58:19,425 - root - INFO - CF Training: Epoch 0001 Iter 0039 / 0638 | Time 12.6s | Iter Loss 0.0274 | Iter Mean Loss 0.0258
2022-06-09 09:58:32,171 - root - INFO - CF Training: Epoch 0001 Iter 0040 / 0638 | Time 12.7s | Iter Loss 0.0267 | Iter Mean Loss 0.0258
2022-06-09 09:58:44,275 - root - INFO - CF Training: Epoch 0001 Iter 0041 / 0638 | Time 12.1s | Iter Loss 0.0295 | Iter Mean Loss 0.0259
2022-06-09 09:58:58,101 - root - INFO - CF Training: Epoch 0001 Iter 0042 / 0638 | Time 13.8s | Iter Loss 0.0291 | Iter Mean Loss 0.0260
2022-06-09 09:59:12,314 - root - INFO - CF Training: Epoch 0001 Iter 0043 / 0638 | Time 14.2s | Iter Loss 0.0257 | Iter Mean Loss 0.0259
2022-06-09 09:59:26,463 - root - INFO - CF Training: Epoch 0001 Iter 0044 / 0638 | Time 14.1s | Iter Loss 0.0262 | Iter Mean Loss 0.0260
2022-06-09 09:59:39,362 - root - INFO - CF Training: Epoch 0001 Iter 0045 / 0638 | Time 12.9s | Iter Loss 0.0260 | Iter Mean Loss 0.0260
2022-06-09 09:59:56,023 - root - INFO - CF Training: Epoch 0001 Iter 0046 / 0638 | Time 16.7s | Iter Loss 0.0329 | Iter Mean Loss 0.0261
2022-06-09 10:00:16,666 - root - INFO - CF Training: Epoch 0001 Iter 0047 / 0638 | Time 20.6s | Iter Loss 0.0258 | Iter Mean Loss 0.0261
2022-06-09 10:00:31,520 - root - INFO - CF Training: Epoch 0001 Iter 0048 / 0638 | Time 14.8s | Iter Loss 0.0236 | Iter Mean Loss 0.0260
2022-06-09 10:00:42,555 - root - INFO - CF Training: Epoch 0001 Iter 0049 / 0638 | Time 11.0s | Iter Loss 0.0236 | Iter Mean Loss 0.0260
2022-06-09 10:00:54,635 - root - INFO - CF Training: Epoch 0001 Iter 0050 / 0638 | Time 12.1s | Iter Loss 0.0256 | Iter Mean Loss 0.0260
2022-06-09 10:01:06,774 - root - INFO - CF Training: Epoch 0001 Iter 0051 / 0638 | Time 12.1s | Iter Loss 0.0237 | Iter Mean Loss 0.0259
2022-06-09 10:01:18,273 - root - INFO - CF Training: Epoch 0001 Iter 0052 / 0638 | Time 11.5s | Iter Loss 0.0265 | Iter Mean Loss 0.0260
2022-06-09 10:01:31,464 - root - INFO - CF Training: Epoch 0001 Iter 0053 / 0638 | Time 13.2s | Iter Loss 0.0256 | Iter Mean Loss 0.0259
2022-06-09 10:01:43,082 - root - INFO - CF Training: Epoch 0001 Iter 0054 / 0638 | Time 11.6s | Iter Loss 0.0236 | Iter Mean Loss 0.0259
2022-06-09 10:01:54,287 - root - INFO - CF Training: Epoch 0001 Iter 0055 / 0638 | Time 11.2s | Iter Loss 0.0266 | Iter Mean Loss 0.0259
2022-06-09 10:02:07,123 - root - INFO - CF Training: Epoch 0001 Iter 0056 / 0638 | Time 12.8s | Iter Loss 0.0277 | Iter Mean Loss 0.0259
2022-06-09 10:02:18,265 - root - INFO - CF Training: Epoch 0001 Iter 0057 / 0638 | Time 11.1s | Iter Loss 0.0227 | Iter Mean Loss 0.0259
2022-06-09 10:02:29,031 - root - INFO - CF Training: Epoch 0001 Iter 0058 / 0638 | Time 10.8s | Iter Loss 0.0236 | Iter Mean Loss 0.0259
2022-06-09 10:02:41,035 - root - INFO - CF Training: Epoch 0001 Iter 0059 / 0638 | Time 12.0s | Iter Loss 0.0254 | Iter Mean Loss 0.0258
2022-06-09 10:02:55,985 - root - INFO - CF Training: Epoch 0001 Iter 0060 / 0638 | Time 14.9s | Iter Loss 0.0253 | Iter Mean Loss 0.0258
2022-06-09 10:03:14,902 - root - INFO - CF Training: Epoch 0001 Iter 0061 / 0638 | Time 18.9s | Iter Loss 0.0251 | Iter Mean Loss 0.0258
2022-06-09 10:03:30,216 - root - INFO - CF Training: Epoch 0001 Iter 0062 / 0638 | Time 15.3s | Iter Loss 0.0256 | Iter Mean Loss 0.0258
2022-06-09 10:03:41,790 - root - INFO - CF Training: Epoch 0001 Iter 0063 / 0638 | Time 11.6s | Iter Loss 0.0236 | Iter Mean Loss 0.0258
2022-06-09 10:03:52,484 - root - INFO - CF Training: Epoch 0001 Iter 0064 / 0638 | Time 10.7s | Iter Loss 0.0248 | Iter Mean Loss 0.0258
2022-06-09 10:04:04,240 - root - INFO - CF Training: Epoch 0001 Iter 0065 / 0638 | Time 11.8s | Iter Loss 0.0302 | Iter Mean Loss 0.0258
2022-06-09 10:04:17,718 - root - INFO - CF Training: Epoch 0001 Iter 0066 / 0638 | Time 13.5s | Iter Loss 0.0250 | Iter Mean Loss 0.0258
2022-06-09 10:04:30,557 - root - INFO - CF Training: Epoch 0001 Iter 0067 / 0638 | Time 12.8s | Iter Loss 0.0254 | Iter Mean Loss 0.0258
2022-06-09 10:04:40,016 - root - INFO - CF Training: Epoch 0001 Iter 0068 / 0638 | Time 9.5s | Iter Loss 0.0216 | Iter Mean Loss 0.0258
2022-06-09 10:04:52,232 - root - INFO - CF Training: Epoch 0001 Iter 0069 / 0638 | Time 12.2s | Iter Loss 0.0253 | Iter Mean Loss 0.0258
2022-06-09 10:04:59,470 - root - INFO - CF Training: Epoch 0001 Iter 0070 / 0638 | Time 7.2s | Iter Loss 0.0228 | Iter Mean Loss 0.0257
2022-06-09 10:05:05,999 - root - INFO - CF Training: Epoch 0001 Iter 0071 / 0638 | Time 6.5s | Iter Loss 0.0249 | Iter Mean Loss 0.0257
2022-06-09 10:05:13,393 - root - INFO - CF Training: Epoch 0001 Iter 0072 / 0638 | Time 7.4s | Iter Loss 0.0280 | Iter Mean Loss 0.0257
2022-06-09 10:05:21,894 - root - INFO - CF Training: Epoch 0001 Iter 0073 / 0638 | Time 8.5s | Iter Loss 0.0292 | Iter Mean Loss 0.0258
2022-06-09 10:05:35,143 - root - INFO - CF Training: Epoch 0001 Iter 0074 / 0638 | Time 13.2s | Iter Loss 0.0223 | Iter Mean Loss 0.0257
2022-06-09 10:05:53,391 - root - INFO - CF Training: Epoch 0001 Iter 0075 / 0638 | Time 18.2s | Iter Loss 0.0285 | Iter Mean Loss 0.0258
2022-06-09 10:06:13,278 - root - INFO - CF Training: Epoch 0001 Iter 0076 / 0638 | Time 19.9s | Iter Loss 0.0231 | Iter Mean Loss 0.0257
2022-06-09 10:06:28,480 - root - INFO - CF Training: Epoch 0001 Iter 0077 / 0638 | Time 15.2s | Iter Loss 0.0289 | Iter Mean Loss 0.0258
2022-06-09 10:06:41,710 - root - INFO - CF Training: Epoch 0001 Iter 0078 / 0638 | Time 13.2s | Iter Loss 0.0262 | Iter Mean Loss 0.0258
2022-06-09 10:06:53,852 - root - INFO - CF Training: Epoch 0001 Iter 0079 / 0638 | Time 12.1s | Iter Loss 0.0263 | Iter Mean Loss 0.0258
2022-06-09 10:07:09,084 - root - INFO - CF Training: Epoch 0001 Iter 0080 / 0638 | Time 15.2s | Iter Loss 0.0245 | Iter Mean Loss 0.0258
2022-06-09 10:07:26,263 - root - INFO - CF Training: Epoch 0001 Iter 0081 / 0638 | Time 17.2s | Iter Loss 0.0272 | Iter Mean Loss 0.0258
2022-06-09 10:07:38,573 - root - INFO - CF Training: Epoch 0001 Iter 0082 / 0638 | Time 12.3s | Iter Loss 0.0202 | Iter Mean Loss 0.0257
2022-06-09 10:07:49,633 - root - INFO - CF Training: Epoch 0001 Iter 0083 / 0638 | Time 11.1s | Iter Loss 0.0264 | Iter Mean Loss 0.0257
2022-06-09 10:08:00,291 - root - INFO - CF Training: Epoch 0001 Iter 0084 / 0638 | Time 10.7s | Iter Loss 0.0232 | Iter Mean Loss 0.0257
2022-06-09 10:08:11,897 - root - INFO - CF Training: Epoch 0001 Iter 0085 / 0638 | Time 11.6s | Iter Loss 0.0217 | Iter Mean Loss 0.0256
2022-06-09 10:08:23,199 - root - INFO - CF Training: Epoch 0001 Iter 0086 / 0638 | Time 11.3s | Iter Loss 0.0236 | Iter Mean Loss 0.0256
2022-06-09 10:08:33,662 - root - INFO - CF Training: Epoch 0001 Iter 0087 / 0638 | Time 10.5s | Iter Loss 0.0238 | Iter Mean Loss 0.0256
2022-06-09 10:08:44,430 - root - INFO - CF Training: Epoch 0001 Iter 0088 / 0638 | Time 10.8s | Iter Loss 0.0229 | Iter Mean Loss 0.0256
2022-06-09 10:09:01,880 - root - INFO - CF Training: Epoch 0001 Iter 0089 / 0638 | Time 17.4s | Iter Loss 0.0261 | Iter Mean Loss 0.0256
2022-06-09 10:09:18,689 - root - INFO - CF Training: Epoch 0001 Iter 0090 / 0638 | Time 16.8s | Iter Loss 0.0289 | Iter Mean Loss 0.0256
2022-06-09 10:09:29,743 - root - INFO - CF Training: Epoch 0001 Iter 0091 / 0638 | Time 11.1s | Iter Loss 0.0241 | Iter Mean Loss 0.0256
2022-06-09 10:09:41,926 - root - INFO - CF Training: Epoch 0001 Iter 0092 / 0638 | Time 12.2s | Iter Loss 0.0234 | Iter Mean Loss 0.0256
2022-06-09 10:09:53,045 - root - INFO - CF Training: Epoch 0001 Iter 0093 / 0638 | Time 11.1s | Iter Loss 0.0219 | Iter Mean Loss 0.0255
2022-06-09 10:10:03,567 - root - INFO - CF Training: Epoch 0001 Iter 0094 / 0638 | Time 10.5s | Iter Loss 0.0274 | Iter Mean Loss 0.0256
2022-06-09 10:10:14,362 - root - INFO - CF Training: Epoch 0001 Iter 0095 / 0638 | Time 10.8s | Iter Loss 0.0278 | Iter Mean Loss 0.0256
2022-06-09 10:10:26,160 - root - INFO - CF Training: Epoch 0001 Iter 0096 / 0638 | Time 11.8s | Iter Loss 0.0277 | Iter Mean Loss 0.0256
2022-06-09 10:10:37,951 - root - INFO - CF Training: Epoch 0001 Iter 0097 / 0638 | Time 11.8s | Iter Loss 0.0239 | Iter Mean Loss 0.0256
2022-06-09 10:10:51,141 - root - INFO - CF Training: Epoch 0001 Iter 0098 / 0638 | Time 13.2s | Iter Loss 0.0246 | Iter Mean Loss 0.0256
2022-06-09 10:11:02,586 - root - INFO - CF Training: Epoch 0001 Iter 0099 / 0638 | Time 11.4s | Iter Loss 0.0262 | Iter Mean Loss 0.0256
2022-06-09 10:11:15,560 - root - INFO - CF Training: Epoch 0001 Iter 0100 / 0638 | Time 13.0s | Iter Loss 0.0234 | Iter Mean Loss 0.0256
2022-06-09 10:11:27,779 - root - INFO - CF Training: Epoch 0001 Iter 0101 / 0638 | Time 12.2s | Iter Loss 0.0281 | Iter Mean Loss 0.0256
2022-06-09 10:11:39,811 - root - INFO - CF Training: Epoch 0001 Iter 0102 / 0638 | Time 12.0s | Iter Loss 0.0261 | Iter Mean Loss 0.0256
2022-06-09 10:11:54,662 - root - INFO - CF Training: Epoch 0001 Iter 0103 / 0638 | Time 14.8s | Iter Loss 0.0319 | Iter Mean Loss 0.0257
2022-06-09 10:12:12,677 - root - INFO - CF Training: Epoch 0001 Iter 0104 / 0638 | Time 18.0s | Iter Loss 0.0243 | Iter Mean Loss 0.0256
2022-06-09 10:12:26,935 - root - INFO - CF Training: Epoch 0001 Iter 0105 / 0638 | Time 14.3s | Iter Loss 0.0252 | Iter Mean Loss 0.0256
2022-06-09 10:12:38,966 - root - INFO - CF Training: Epoch 0001 Iter 0106 / 0638 | Time 12.0s | Iter Loss 0.0239 | Iter Mean Loss 0.0256
2022-06-09 10:12:50,306 - root - INFO - CF Training: Epoch 0001 Iter 0107 / 0638 | Time 11.3s | Iter Loss 0.0242 | Iter Mean Loss 0.0256
2022-06-09 10:13:01,738 - root - INFO - CF Training: Epoch 0001 Iter 0108 / 0638 | Time 11.4s | Iter Loss 0.0230 | Iter Mean Loss 0.0256
2022-06-09 10:13:12,970 - root - INFO - CF Training: Epoch 0001 Iter 0109 / 0638 | Time 11.2s | Iter Loss 0.0219 | Iter Mean Loss 0.0255
2022-06-09 10:13:24,940 - root - INFO - CF Training: Epoch 0001 Iter 0110 / 0638 | Time 12.0s | Iter Loss 0.0278 | Iter Mean Loss 0.0256
2022-06-09 10:13:36,369 - root - INFO - CF Training: Epoch 0001 Iter 0111 / 0638 | Time 11.4s | Iter Loss 0.0234 | Iter Mean Loss 0.0255
2022-06-09 10:13:48,581 - root - INFO - CF Training: Epoch 0001 Iter 0112 / 0638 | Time 12.2s | Iter Loss 0.0234 | Iter Mean Loss 0.0255
2022-06-09 10:14:01,100 - root - INFO - CF Training: Epoch 0001 Iter 0113 / 0638 | Time 12.5s | Iter Loss 0.0243 | Iter Mean Loss 0.0255
2022-06-09 10:14:15,570 - root - INFO - CF Training: Epoch 0001 Iter 0114 / 0638 | Time 14.5s | Iter Loss 0.0267 | Iter Mean Loss 0.0255
2022-06-09 10:14:26,718 - root - INFO - CF Training: Epoch 0001 Iter 0115 / 0638 | Time 11.1s | Iter Loss 0.0253 | Iter Mean Loss 0.0255
2022-06-09 10:14:41,250 - root - INFO - CF Training: Epoch 0001 Iter 0116 / 0638 | Time 14.5s | Iter Loss 0.0242 | Iter Mean Loss 0.0255
2022-06-09 10:14:52,962 - root - INFO - CF Training: Epoch 0001 Iter 0117 / 0638 | Time 11.7s | Iter Loss 0.0222 | Iter Mean Loss 0.0255
2022-06-09 10:15:11,393 - root - INFO - CF Training: Epoch 0001 Iter 0118 / 0638 | Time 18.4s | Iter Loss 0.0243 | Iter Mean Loss 0.0255
2022-06-09 10:15:26,846 - root - INFO - CF Training: Epoch 0001 Iter 0119 / 0638 | Time 15.4s | Iter Loss 0.0278 | Iter Mean Loss 0.0255
2022-06-09 10:15:37,139 - root - INFO - CF Training: Epoch 0001 Iter 0120 / 0638 | Time 10.3s | Iter Loss 0.0261 | Iter Mean Loss 0.0255
2022-06-09 10:15:47,826 - root - INFO - CF Training: Epoch 0001 Iter 0121 / 0638 | Time 10.7s | Iter Loss 0.0266 | Iter Mean Loss 0.0255
2022-06-09 10:16:01,887 - root - INFO - CF Training: Epoch 0001 Iter 0122 / 0638 | Time 14.1s | Iter Loss 0.0229 | Iter Mean Loss 0.0255
2022-06-09 10:16:13,667 - root - INFO - CF Training: Epoch 0001 Iter 0123 / 0638 | Time 11.8s | Iter Loss 0.0216 | Iter Mean Loss 0.0255
2022-06-09 10:16:25,272 - root - INFO - CF Training: Epoch 0001 Iter 0124 / 0638 | Time 11.6s | Iter Loss 0.0247 | Iter Mean Loss 0.0255

****************************************************
* ecfkg model last log amazon-book

****************************************************
6 | Time 0.3s | Iter Loss 1.1896 | Iter Mean Loss 1.1781
2022-06-09 17:43:20,917 - root - INFO - KG Training: Epoch 0010 Iter 3126 / 3136 | Time 0.6s | Iter Loss 1.1896 | Iter Mean Loss 1.1781
2022-06-09 17:43:21,247 - root - INFO - KG Training: Epoch 0010 Iter 3127 / 3136 | Time 0.3s | Iter Loss 1.1675 | Iter Mean Loss 1.1781
2022-06-09 17:43:21,576 - root - INFO - KG Training: Epoch 0010 Iter 3128 / 3136 | Time 0.3s | Iter Loss 1.1679 | Iter Mean Loss 1.1781
2022-06-09 17:43:21,991 - root - INFO - KG Training: Epoch 0010 Iter 3129 / 3136 | Time 0.4s | Iter Loss 1.1870 | Iter Mean Loss 1.1781
2022-06-09 17:43:22,355 - root - INFO - KG Training: Epoch 0010 Iter 3130 / 3136 | Time 0.4s | Iter Loss 1.1800 | Iter Mean Loss 1.1781
2022-06-09 17:43:22,723 - root - INFO - KG Training: Epoch 0010 Iter 3131 / 3136 | Time 0.4s | Iter Loss 1.2090 | Iter Mean Loss 1.1781
2022-06-09 17:43:23,066 - root - INFO - KG Training: Epoch 0010 Iter 3132 / 3136 | Time 0.3s | Iter Loss 1.1738 | Iter Mean Loss 1.1781
2022-06-09 17:43:23,818 - root - INFO - KG Training: Epoch 0010 Iter 3133 / 3136 | Time 0.8s | Iter Loss 1.1676 | Iter Mean Loss 1.1781
2022-06-09 17:43:24,137 - root - INFO - KG Training: Epoch 0010 Iter 3134 / 3136 | Time 0.3s | Iter Loss 1.1679 | Iter Mean Loss 1.1781
2022-06-09 17:43:24,697 - root - INFO - KG Training: Epoch 0010 Iter 3135 / 3136 | Time 0.6s | Iter Loss 1.1698 | Iter Mean Loss 1.1781
2022-06-09 17:43:25,256 - root - INFO - KG Training: Epoch 0010 Iter 3136 / 3136 | Time 0.6s | Iter Loss 1.1732 | Iter Mean Loss 1.1781
2022-06-09 17:43:25,257 - root - INFO - KG Training: Epoch 0010 Total Iter 3136 | Total Time 1209.0s | Iter Mean Loss 1.1781




