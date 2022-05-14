#!/usr/bin/bash

printf -v date '%(%Y-%m-%d)T' -1
outdir=evaluation_$date

# COCOstuff restricted to UnRel compatible classes
python test_recognition.py --model_yaml backup/unrel_compatible/model.yaml --checkpoint backup/unrel_compatible/model_final.pth --dataset COCOSTUFF_UNREL_COMPATIBLE  --record_individual_scores --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/unrel_compatible_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/unrel_compatible_accuracies.json
rm ${outdir}/evaluation_*.json
rm ${outdir}/instances_predictions.pth
rm ${outdir}/coco_instances_results.json

# UnRel
python test_recognition.py --model_yaml backup/unrel_compatible/model.yaml --checkpoint backup/unrel_compatible/model_final.pth --dataset UNREL  --record_individual_scores --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/unrel_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/unrel_accuracies.json
rm ${outdir}/evaluation_*.json
rm ${outdir}/instances_predictions.pth
rm ${outdir}/coco_instances_results.json

# COCOstuff restricted to virtualhome compatible classes
python test_recognition.py --model_yaml backup/virtualhome_compatible/model.yaml --checkpoint backup/virtualhome_compatible/model_final.pth --dataset COCOSTUFF_VIRTUALHOME_COMPATIBLE  --record_individual_scores --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/virtualhome_compatible_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/virtualhome_compatible_accuracies.json
rm ${outdir}/evaluation_*.json
rm ${outdir}/instances_predictions.pth
rm ${outdir}/coco_instances_results.json

# virtualhome IC
python test_recognition.py --model_yaml backup/virtualhome_compatible/model.yaml --checkpoint backup/virtualhome_compatible/model_final.pth --dataset VIRTUALHOME --record_individual_scores --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/virtualhome_IC_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/virtualhome_IC_accuracies.json
rm ${outdir}/evaluation_*.json
rm ${outdir}/instances_predictions.pth
rm ${outdir}/coco_instances_results.json

# virtualhome gravity
python test_recognition.py --model_yaml backup/virtualhome_compatible/model.yaml --checkpoint backup/virtualhome_compatible/model_final.pth --dataset VIRTUALHOME_GRAVITY --record_individual_scores --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/virtualhome_gravity_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/virtualhome_gravity_accuracies.json
rm ${outdir}/evaluation_*.json
rm ${outdir}/instances_predictions.pth
rm ${outdir}/coco_instances_results.json

# virtualhome anomaly
python test_recognition.py --model_yaml backup/virtualhome_compatible/model.yaml --checkpoint backup/virtualhome_compatible/model_final.pth --dataset VIRTUALHOME_ANOMALY --record_individual_scores --outdir $outdir
mv ${outdir}/individual_scores.json ${outdir}/virtualhome_anomaly_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/virtualhome_anomaly_accuracies.json
rm ${outdir}/evaluation_*.json
rm ${outdir}/instances_predictions.pth
rm ${outdir}/coco_instances_results.json

