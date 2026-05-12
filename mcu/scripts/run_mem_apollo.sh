python cal_rw.py --model ff --in_features 784 --width 64
python cal_rw.py --model ff --in_features 300 --width 128
python cal_rw.py --model ff --in_features 761 --width 64

python cal_rw.py --model fff --in_features 784 --depth 2 --leaf_width 16
python cal_rw.py --model fff --in_features 300 --depth 2 --leaf_width 32
python cal_rw.py --model fff --in_features 761 --depth 2 --leaf_width 16

python cal_rw.py --model vfff --in_features 784 --depth 2 --page_size 400 --pages_per_dim 29 --leaf_width 16
python cal_rw.py --model vfff --in_features 300 --depth 2 --page_size 10 --pages_per_dim 800 --leaf_width 32
python cal_rw.py --model vfff --in_features 761 --depth 2 --page_size 100 --pages_per_dim 80 --leaf_width 16
