#!/bin/bash

prefix=debug5-sampled
# KET

python ./sample4Annotators-2sims.py final-glove.840B-simonly99.tsv ../../data/classInfo.tsv KET 11 2 4 1 simonly > ${prefix}-ket-11_2_4_1_simonly.tsv 
cut -f 1-3 < ${prefix}-ket-11_2_4_1_simonly.tsv | head -100 | shuf  --random-source=final-glove.840B-simonly99.tsv > ${prefix}-ket_11_2_4_1_simonly-4anns.tsv

python ./sample4Annotators-2sims.py final-glove.840B-simonly99.tsv ../../data/classInfo.tsv KET 11 2 4 1 simidfonly > ${prefix}-ket-11_2_4_1_simidfonly.tsv
cut -f 1-3 < ${prefix}-ket-11_2_4_1_simidfonly.tsv | head -100 | shuf  --random-source=final-glove.840B-simonly99.tsv > ${prefix}-ket_11_2_4_1_simidfonly-4anns.tsv

python ./sample4Annotators-2sims.py final-glove.840B-simonly99.tsv ../../data/classInfo.tsv KET 11 2 4 1 simidfstr > ${prefix}-ket-11_2_4_1_simidfstr.tsv
cut -f 1-3 < ${prefix}-ket-11_2_4_1_simidfstr.tsv | head -100 | shuf  --random-source=final-glove.840B-simonly99.tsv > ${prefix}-ket_11_2_4_1_simidfstr-4anns.tsv

# SGC

python ./sample4Annotators-2sims.py final-glove.840B-simonly99.tsv ../../data/classInfo.tsv SGC 11 2 4 1 simonly > ${prefix}-sgc-11_2_4_1_simonly.tsv
cut -f 1-3 < ${prefix}-sgc-11_2_4_1_simonly.tsv | head -100 | shuf  --random-source=final-glove.840B-simonly99.tsv > ${prefix}-sgc_11_2_4_1_simonly-4anns.tsv

python ./sample4Annotators-2sims.py final-glove.840B-simonly99.tsv ../../data/classInfo.tsv SGC 11 2 4 1 simidfonly > ${prefix}-sgc-11_2_4_1_simidfonly.tsv
cut -f 1-3 < ${prefix}-sgc-11_2_4_1_simidfonly.tsv | head -100 | shuf  --random-source=final-glove.840B-simonly99.tsv > ${prefix}-sgc_11_2_4_1_simidfonly-4anns.tsv

python ./sample4Annotators-2sims.py final-glove.840B-simonly99.tsv ../../data/classInfo.tsv SGC 11 2 4 1 simidfstr > ${prefix}-sgc-11_2_4_1_simidfstr.tsv
cut -f 1-3 < ${prefix}-sgc-11_2_4_1_simidfstr.tsv | head -100 | shuf  --random-source=final-glove.840B-simonly99.tsv > ${prefix}-sgc_11_2_4_1_simidfstr-4anns.tsv

