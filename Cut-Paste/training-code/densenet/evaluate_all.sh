#!/usr/bin/bash

printf -v date '%(%Y-%m-%d)T' -1
outdir=evaluation_$date

# COCOstuff restricted to UnRel compatible classes
python test.py --checkpoint backup/unrel_compatible/checkpoint_10.pth --annotations_file /media/data/philipp_data/COCOstuff/annotations_UnRel_compatible/val.json --image_dir /media/data/philipp_data/COCOstuff/images/val/ --output_dir $outdir --record_individual_scores --num_classes 33
mv ${outdir}/individual_scores.json ${outdir}/unrel_compatible_individual_scores.json
mv ${outdir}/test_accuracies.json ${outdir}/unrel_compatible_accuracies.json

# UnRel
python test.py --checkpoint backup/unrel_compatible/checkpoint_10.pth --annotations_file /media/data/philipp_data/UnRel_test/annotations/annotations.json --image_dir /media/data/philipp_data/UnRel_test/images --output_dir $outdir --record_individual_scores --num_classes 33
mv ${outdir}/individual_scores.json ${outdir}/unrel_individual_scores.json
mv ${outdir}/test_accuracies.json ${outdir}/unrel_accuracies.json

# COCOstuff restricted to virtualhome compatible classes
python test.py --checkpoint backup/virtualhome_compatible/checkpoint_10.tar --annotations_file /media/data/philipp_data/COCOstuff/annotations_virtualhome_compatible/val.json --image_dir /media/data/philipp_data/COCOstuff/images/val/ --output_dir $outdir --record_individual_scores --num_classes 15
mv ${outdir}/individual_scores.json ${outdir}/virtualhome_compatible_individual_scores.json
mv ${outdir}/test_accuracies.json ${outdir}/virtualhome_compatible_accuracies.json

# virtualhome IC
python test.py --checkpoint backup/virtualhome_compatible/checkpoint_10.tar --annotations_file /media/data/philipp_data/virtualhome/virtualhome_IC/annotations.json --image_dir /media/data/philipp_data/virtualhome/virtualhome_IC --output_dir $outdir --record_individual_scores --num_classes 15
mv ${outdir}/individual_scores.json ${outdir}/virtualhome_IC_individual_scores.json
mv ${outdir}/test_accuracies.json ${outdir}/virtualhome_IC_accuracies.json

# virtualhome gravity
python test.py --checkpoint backup/virtualhome_compatible/checkpoint_10.tar --annotations_file /media/data/philipp_data/virtualhome/virtualhome_gravity/annotations.json --image_dir /media/data/philipp_data/virtualhome/virtualhome_gravity --output_dir $outdir --record_individual_scores --num_classes 15
mv ${outdir}/individual_scores.json ${outdir}/virtualhome_gravity_individual_scores.json
mv ${outdir}/test_accuracies.json ${outdir}/virtualhome_gravity_accuracies.json

# virtualhome anomaly
python test.py --checkpoint backup/virtualhome_compatible/checkpoint_10.tar --annotations_file /media/data/philipp_data/virtualhome/virtualhome_anomaly/annotations.json --image_dir /media/data/philipp_data/virtualhome/virtualhome_anomaly --output_dir $outdir --record_individual_scores --num_classes 15
mv ${outdir}/individual_scores.json ${outdir}/virtualhome_anomaly_individual_scores.json
mv ${outdir}/test_accuracies.json ${outdir}/virtualhome_anomaly_accuracies.json
