import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_percent', type=float, default=0.65)
    parser.add_argument('--val_percent', type=float, default=0.15)
    parser.add_argument('--category', type=str, default='safety',
                        help='onf of [beautiful, boring, depressing, '
                             'lively, safety, wealthy]')
    parser.add_argument('--file_path', type=str,
                        default='F:\\data\\PlacePulse\\annos.csv')
    parser.add_argument('--target_path', type=str,
                        default='F:\\data\\PlacePulse')
    parser.add_argument('--include_equal', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.val_percent + args.train_percent >= 1.0:
        raise ValueError('sum of train_percent & val_percent must < 1')

    df = pd.read_csv(args.file_path)
    category = args.category
    if category is not None:
        df = df[df.category == category]
    not_equal_df = df[df.winner != 'equal']

    num_train = int(len(not_equal_df) * args.train_percent)
    num_val = int(len(not_equal_df) * args.val_percent)
    num_test = len(not_equal_df) - num_val - num_train

    if num_test <= 0 or num_val <= 0 or num_train <= 0:
        raise ValueError('num_train/num_val/num_test must be positive')

    not_equal_df = not_equal_df.sample(frac=1.0)  # shuffle
    train_df = not_equal_df[:num_train]
    val_df = not_equal_df[num_train:-num_test]
    test_df = not_equal_df[-num_test:]

    if args.include_equal:
        equal_df = df[df.winner == 'equal']
        train_df = pd.concat([train_df, equal_df])

    if category is None:
        category = 'total'
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)
    if args.include_equal:
        base_file_name = '{}_{}_equal.csv'
    else:
        base_file_name = '{}_{}.csv'
    target_format = os.path.join(args.target_path, base_file_name)

    train_df.to_csv(target_format.format(
        category, 'train'), index=False)
    val_df.to_csv(target_format.format(category, 'val'), index=False)
    test_df.to_csv(target_format.format(
        category, 'test'), index=False)
