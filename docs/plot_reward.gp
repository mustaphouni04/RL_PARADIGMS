set terminal pngcairo size 1200,800 enhanced font 'Arial,14'
set output 'mean_reward_vs_frames.png'

set title "Mean Reward vs Training Frames"
set xlabel "Frames"
set ylabel "Mean Reward"
set grid

set key left top

set cblabel "Epsilon"
set palette defined (0 "red", 1 "blue")

plot 'train_data_n_step.txt' using 1:2:3 with points pt 7 ps 1.5 palette title "Mean Reward (color = epsilon)" , \
     'train_data_n_step.txt' using 1:2 with lines lw 2 title "Reward Trend"
