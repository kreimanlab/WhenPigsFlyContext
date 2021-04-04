#!/usr/bin/bash

printf -v date '%(%Y-%m-%d)T' -1
outdir=evaluation_$date

# Congruent vs Incongruent Exp I
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_I  --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/exp_I_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_I_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_I_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_I_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp J
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_J  --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/exp_J_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_J_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_J_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_J_coco_instances_results.json
rm ${outdir}/evaluation_*.json

