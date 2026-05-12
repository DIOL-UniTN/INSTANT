import argparse

def cal_vfff(in_features, depth, leaf_width, page_size, pages_per_dim):
    out_features = 10
    read  = leaf_width * in_features * depth + pages_per_dim + pages_per_dim * page_size * 2 * depth + leaf_width * out_features * 2
    write = leaf_width + out_features
    return read, write

def cal_fff(in_features, depth, leaf_width):
    out_features = 10
    read = leaf_width * in_features * depth + leaf_width * in_features * 2 + leaf_width * out_features * 2
    write = leaf_width + out_features
    return read, write

def cal_ff(in_features, width):
    out_features = 10
    read = in_features * width + width * out_features * 2
    write = width + out_features
    return read, write

def main(args):
    if args.model == 'vfff':
        read, write = cal_vfff(args.in_features, args.depth, args.leaf_width, args.page_size, args.pages_per_dim)
    elif args.model == 'fff':
        read, write = cal_fff(args.in_features, args.depth, args.leaf_width)
    elif args.model == 'ff':
        read, write = cal_ff(args.in_features, args.width)
    # print(f"Read: {read}, Write: {write}")

    f = 96000000
    p_write = 7.2 * 10**-3
    p_read = 7.3* 10**-3

    print(f"{args.model}")
    e_total = (1/f) * read * (p_read + p_write)
    print(f"Energy {e_total}")
    # print(f"Energy Read: {e_read}, Write: {e_write}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate read/write for different models')
    parser.add_argument('--model', type=str, choices=['vfff', 'fff', 'ff'], help='Model type to calculate read/write for')
    parser.add_argument('--in_features', type=int, default=128, help='Number of input features')
    parser.add_argument('--depth', type=int, default=4, help='Depth of the model')
    parser.add_argument('--leaf_width', type=int, default=64, help='Width of the leaf nodes')
    parser.add_argument('--page_size', type=int, default=256, help='Size of a page')
    parser.add_argument('--pages_per_dim', type=int, default=4, help='Number of pages per dimension')
    parser.add_argument('--width', type=int, default=128, help='Width of the model for FF')
    args = parser.parse_args()
    main(args)
