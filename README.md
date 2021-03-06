# BTransX
The implementation of BTransE and BTransR for Cross-Lingual Taxonomy Alignment.
## Data input
Datasets are required in the following format, all entities and relations can be represented by ids. We call id the index of entities and relations.

ZH triples: [ZH_h, ZH_t, ZH_r], where ZH_h, ZH_t, ZH_r is the head entity, tail entity, relation of a Chinese triple respectively.

EN triples: [EN_h, EN_t, EN_r], where EN_h, EN_t, EN_r is the head entity, tail entity, relation of an English triple respectively.

aligned triples: [[ZH_h, ZH_t, ZH_r, ZH_h_index, ZH_t_index, ZH_r_index, EN_h, EN_t, EN_r, EN_h_index, EN_t_index, EN_r_index], ..., ].

For example, [["Rockstar", "Rockstar游戏公司", "公司名称", 15234, 15233, 688, "Rockstar_Games", "Rockstar Games, Inc.", "companyName", 36423, 36420, 486], ..., ].

## Data output
For BTransE, the program will output two matrices, i.e. matrix for head translation and matrix for tail translation.

For BTransR, the program will output N matrices, where N is the number of relations in English.
