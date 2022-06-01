ps -ef | grep Learner | grep -v grep | awk '{print "kill "$2}' | sh
ps -ef | grep plasma | grep -v grep | awk '{print "kill "$2}' | sh
ps -ef | grep tensorboard | grep -v grep | awk '{print "kill "$2}' | sh
ps -ef | grep Worker | grep -v grep | awk '{print "kill "$2}' | sh
rm -rf logs
rm -rf Plasma_server/plasma/multi_point_heterogeneous_policy
# rm -rf Exp/Model/model_pool/*
rm -rf Worker/Download_model/*
# -------- 删除worker端保存下来的数据 ----------
# rm -rf Data_saved/multi_point_heterogeneous_policy
rm -rf nohup.out
# rm -rf Exp

ps -ef | grep multiprocessing.spawn | grep -v grep | awk '{print "kill "$2}' | sh