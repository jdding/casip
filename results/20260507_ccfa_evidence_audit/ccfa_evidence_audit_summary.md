# CCF-A Evidence Audit

## Tmall Alternate Validation

|split|validation_users|feasible_policies|selected_policy|net@50_val|net@100_val|hit@50_test|hit@100_test|net@50_test|net@100_test|ratio@100_test|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|validation_1|1648|102|cat_brand__sem50__n_cat_brand_le_20|11|29|24920|26080|67|1070|0.039|
|validation_2|1545|111|cat_brand__sem50__n_cat_brand_le_20|16|29|24920|26080|67|1070|0.039|
|validation_3|1589|131|cat_brand__sem50__n_brand_le_20|10|31|24797|26075|-56|1065|0.049|

## Tmall Main Slices

|group|slice|users|base@100|casp@100|net@100|gross@100|cannibal@100|ratio@100|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|target_oov_user_history|all|45574|25010|26080|1070|1114|44|0.039|
|target_oov_user_history|True|33037|12483|13569|1086|1112|26|0.023|
|target_oov_user_history|False|12537|12527|12511|-16|2|18|9.000|
|target_oov_pre_history|all|45574|25010|26080|1070|1114|44|0.039|
|target_oov_pre_history|True|45330|24766|25837|1071|1114|43|0.039|
|target_oov_pre_history|False|244|244|243|-1|0|1||
|opened|all|45574|25010|26080|1070|1114|44|0.039|
|opened|1|42531|22311|23381|1070|1114|44|0.039|
|opened|0|3043|2699|2699|0|0|0||

## Tmall Bootstrap

|metric|observed|ci_low|ci_high|p_boot_le_zero|
|---|---:|---:|---:|---:|
|net@10|101|82.000|120.000|0.000|
|net@50|67|-26.000|156.025|0.077|
|net@100|1070|1008.000|1135.025|0.000|
|net@500|1109|1044.000|1176.000|0.000|

## REES46 Slices

|group|slice|users|base@100|casp@100|net@100|gross@100|cannibal@100|ratio@100|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|target_oov_user_history|all|16954|6392|6438|46|83|37|0.446|
|target_oov_user_history|True|15383|4925|4973|48|83|35|0.422|
|target_oov_user_history|False|1571|1467|1465|-2|0|2||
|target_oov_pre_history|all|16954|6392|6438|46|83|37|0.446|
|target_oov_pre_history|True|16576|6115|6163|48|83|35|0.422|
|target_oov_pre_history|False|378|277|275|-2|0|2||
|target_right_new_global_pre|all|16954|6392|6438|46|83|37|0.446|
|target_right_new_global_pre|False|12591|4991|5036|45|78|33|0.423|
|target_right_new_global_pre|True|4363|1401|1402|1|5|4|0.800|
|opened|all|16954|6392|6438|46|83|37|0.446|
|opened|1|8855|3573|3619|46|83|37|0.446|
|opened|0|8099|2819|2819|0|0|0||

## REES46 Bootstrap

|metric|observed|ci_low|ci_high|p_boot_le_zero|
|---|---:|---:|---:|---:|
|net@10|0|0.000|0.000|1.000|
|net@50|31|7.000|55.000|0.006|
|net@100|46|23.975|67.000|0.000|
|net@500|24|13.000|35.000|0.000|
