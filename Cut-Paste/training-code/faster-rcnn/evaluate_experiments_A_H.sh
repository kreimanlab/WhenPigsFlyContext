#!/usr/bin/bash

printf -v date '%(%Y-%m-%d)T' -1
outdir=evaluation_$date

# Congruent vs Incongruent Exp A
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_A  --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/exp_A_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_A_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_A_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_A_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp H
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_H  --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/exp_H_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_H_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_H_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_H_coco_instances_results.json
rm ${outdir}/evaluation_*.json

