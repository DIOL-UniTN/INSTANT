python cal_rw.py --model ff --in_features 784 --width 16
python cal_rw.py --model ff --in_features 300 --width 32
python cal_rw.py --model ff --in_features 761 --width 16

python cal_rw.py --model fff --in_features 784 --depth 2 --leaf_width 4
python cal_rw.py --model fff --in_features 300 --depth 2 --leaf_width 8
python cal_rw.py --model fff --in_features 761 --depth 2 --leaf_width 4

python cal_rw.py --model vfff --in_features 784 --depth 2 --page_size 400 --pages_per_dim 5 --leaf_width 4
python cal_rw.py --model vfff --in_features 300 --depth 2 --page_size 10 --pages_per_dim 200 --leaf_width 8
python cal_rw.py --model vfff --in_features 761 --depth 2 --page_size 100 --pages_per_dim 20 --leaf_width 4
