#!/bin/bash
#export PYTHONPATH=/home/ziqizhang/chase/python/src
export PYTHONPATH=/home/zz/Work/texpr/code
# filter_by_sim=False
# window=5
# topn=100
# sim_score_files=/home/zqz/Work/data/texpr/acl_sim/with_dict
# sys_folder=/home/zqz/Work/data/texpr/word_weights/acl
# ate_alg=0
# ate_terms_outfile=/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/ttf.json
# ate_terms_outfolder=/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/min1
# outfolder=/home/zqz/Work/data/texpr/texpr_output/acl
# in_corpus=/home/zqz/Work/data/jate_data/acl-rd-corpus-2.0/raw_abstract_plain_txt

sim_score_folder=/home/zqz/Work/data/texpr
sys_folders=/home/zqz/Work/data/texpr/word_weights
ate_terms_folder=/home/zqz/Work/data/semrerank/jate_lrec2016
outfolders=/home/zqz/Work/data/texpr/texpr_output
in_corpus_folder=/home/zqz/Work/data/jate_data


#new models proposed in swj. although the descriptor does not show dropout, it is used. see code add_skipped_conv1d_submodel_other_layers
SETTINGS=(
"filter_by_sim=True window=5 topn=0.999 min_sim=0.5 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.5 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.55 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.55 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.6 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.6 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.65 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.65 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.7 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.7 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.75 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.75 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.8 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.8 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.85 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=5 topn=0.999 min_sim=0.85 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.5 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.5 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.55 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.55 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.6 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.6 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.65 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.65 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.7 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.7 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.75 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.75 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.8 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.8 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.85 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=10 topn=0.999 min_sim=0.85 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.5 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.5 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.55 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.55 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.6 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.6 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.65 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.65 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.7 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.7 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.75 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.75 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.8 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.8 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.85 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2/min1 outfolder=$outfolders/acl/jate in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
"filter_by_sim=True window=100 topn=0.999 min_sim=0.85 sim_score_files=$sim_score_folder/acl_sim/with_dict sys_folder=$sys_folders/acl ate_terms_outfile=$ate_terms_folder/aclrd_ver2_atr4s/ttf.json ate_terms_outfolder=$ate_terms_folder/aclrd_ver2_atr4s/min1 outfolder=$outfolders/acl/atr4s in_corpus=$in_corpus_folder/acl-rd-corpus-2.0/raw_abstract_plain_txt" 
)

IFS=""

echo ${#SETTINGS[@]}
c=0
for s in ${SETTINGS[*]}
do
    printf '\n'
    c=$[$c +1]
    echo ">>> Start the following setting at $(date): "
    echo $c
    line="\t${s}"
    echo -e $line
    python3 -m texpr.texpr_main ${s}
    echo "<<< completed at $(date): "
done



