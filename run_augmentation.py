from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='path to configuration file')

    
    