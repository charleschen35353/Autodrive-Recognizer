yourfilenames=`ls /home/pacowong/research/datasets/hkfi_report_test_sets/test_set_1/damage/*.jpg`
for eachfile in $yourfilenames
do
   echo $eachfile
   python demo_model.py $eachfile
done

yourfilenames=`ls /home/pacowong/research/datasets/hkfi_report_test_sets/test_set_1/no_damage/*.jpg`
for eachfile in $yourfilenames
do
   echo $eachfile
   python demo_model.py $eachfile
done

