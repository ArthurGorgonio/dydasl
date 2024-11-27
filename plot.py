import argparse
# import operator
from glob import glob
from os.path import abspath
from subprocess import CalledProcessError, run

import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv


def get_dataset_names(data_path: str) -> set:
    """get dataset names to filter"""
    base_files = glob(f"{data_path}/*general.dat")

    return sorted(base_files)


def generate_cd_diagram(
    dt_name: str,
    pos: list[int],
    lbs: list[str],
    plot_name: str,
    plot_path: str,
    decreasing: bool = False,
):
    cmd = [
        "Rscript",
        "--vanilla",
        "/home/arthur/workspace/cd-plots/cdplot/mainPlot.R",
        "--location",
        abspath(plot_path) + "/",
        "--head",
        "--cd",
        "--only",
        "".join(str(pos)[1:-1].split(" ")),
        "--col",
        "".join(str(lbs)[1:-1].replace("'", "").split(" ")),
        "--width 1000",
        "--height 640",
        "--cex 2.0",
        "--suffix",
        plot_name,
        dt_name,
    ]

    if decreasing:
        cmd.insert(6, "--decreasing")

    print(f"O comando chamado foi:\n{cmd}")
    try:
        run(' '.join(cmd), check=True, shell=True, stdout=True, stderr=True)
    except CalledProcessError as ex:
        print(f"The {dt_name} not run. Probally all columns are equal!\n\n{ex}")
        raise


def create_plot(
    df_: DataFrame,
    colors: dict[str, dict[str, str]],
    title: str,
    plot_name: str,
    plot_path: str,
    resize: bool = False,
):
    if resize:
        plt.tight_layout()
        plt.figure(figsize=(16.8, 4.8))
    else:
        plt.figure(figsize=(6.4, 4.8))

    p = plt.boxplot(df_, labels=df_.axes[1].array, patch_artist=True)

    for patch, color in zip(p["boxes"], colors):
        patch.set_facecolor(color)

    dir_name = plot_path.split("/")

    if "acc" in dir_name or "f1" in dir_name:
        plt.ylim(0, 1)
    elif "kappa" in dir_name:
        plt.ylim(-1, 1)
    elif "size" in dir_name:
        plt.ylim(0, 10)
    plt.title(f"{title}")

    plt.savefig(f"{plot_path}/{plot_name.lower()}.png")
    plt.clf()


def select_columns(df_columns: list[str], approaches: list[str]) -> list[int]:
    return [df_columns.index(pos) for pos in approaches]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="Path to results files")
    parser.add_argument(
        "--decrease",
        "-d",
        action="store_true",
        help="Is the metric is best when results are lowers? (Default 1)"
        "Pass 0 to False. Any value to True.",
    )
    args = parser.parse_args()

    labels = {
        'F-F-E': {
            'color': '#FFC533FF',
            'order': 1,
            'detector': 'fixed threshold',
            'reator': 'exchange',
        },
        'F-N-E': {
            'color': '#D579FFFF',
            'order': 2,
            'detector': 'normal threshold',
            'reator': 'exchange',
        },
        'F-W-E': {
            'color': '#17A2E3FF',
            'order': 3,
            'detector': 'weight threshold',
            'reator': 'exchange',
        },
        'F-S-E': {
            'color': '#19FFB2CC',
            'order': 4,
            'detector': 'statistical threshold',
            'reator': 'exchange',
        },

        'S-F-E': {
            'color': '#FFC533FF',
            'order': 5,
            'detector': 'fixed threshold',
            'reator': 'exchange',
        },
        'S-N-E': {
            'color': '#D579FFFF',
            'order': 6,
            'detector': 'normal threshold',
            'reator': 'exchange',
        },
        'S-W-E': {
            'color': '#17A2E3FF',
            'order': 7,
            'detector': 'weight threshold',
            'reator': 'exchange',
        },
        'S-S-E': {
            'color': '#19FFB2CC',
            'order': 8,
            'detector': 'statistical threshold',
            'reator': 'exchange',
        },

        'D-F-E': {
            'color': '#FFC533FF',
            'order': 9,
            'detector': 'fixed threshold',
            'reator': 'exchange',
        },
        'D-N-E': {
            'color': '#D579FFFF',
            'order': 10,
            'detector': 'normal threshold',
            'reator': 'exchange',
        },
        'D-W-E': {
            'color': '#17A2E3FF',
            'order': 11,
            'detector': 'weight threshold',
            'reator': 'exchange',
        },
        'D-S-E': {
            'color': '#19FFB2CC',
            'order': 12,
            'detector': 'statistical threshold',
            'reator': 'exchange',
        },

        'S-F-P': {
            'color': '#FFC533FF',
            'order': 13,
            'detector': 'fixed threshold',
            'reator': 'pareto',
        },
        'S-N-P': {
            'color': '#D579FFFF',
            'order': 14,
            'detector': 'normal threshold',
            'reator': 'pareto',
        },
        'S-W-P': {
            'color': '#17A2E3FF',
            'order': 15,
            'detector': 'weight threshold',
            'reator': 'pareto',
        },
        'S-S-P': {
            'color': '#19FFB2CC',
            'order': 16,
            'detector': 'statistical threshold',
            'reator': 'pareto',
        },

        'D-F-P': {
            'color': '#FFC533FF',
            'order': 17,
            'detector': 'fixed threshold',
            'reator': 'pareto',
        },
        'D-N-P': {
            'color': '#D579FFFF',
            'order': 18,
            'detector': 'normal threshold',
            'reator': 'pareto',
        },
        'D-W-P': {
            'color': '#17A2E3FF',
            'order': 19,
            'detector': 'weight threshold',
            'reator': 'pareto',
        },
        'D-S-P': {
            'color': '#19FFB2CC',
            'order': 20,
            'detector': 'statistical threshold',
            'reator': 'pareto',
        },

        'S-F-V': {
            'color': '#FFC533FF',
            'order': 21,
            'detector': 'fixed threshold',
            'reator': 'volatile exchange',
        },
        'S-N-V': {
            'color': '#D579FFFF',
            'order': 22,
            'detector': 'normal threshold',
            'reator': 'volatile exchange',
        },
        'S-W-V': {
            'color': '#17A2E3FF',
            'order': 23,
            'detector': 'weight threshold',
            'reator': 'volatile exchange',
        },
        'S-S-V': {
            'color': '#19FFB2CC',
            'order': 24,
            'detector': 'statistical threshold',
            'reator': 'volatile exchange',
        },

        'D-F-V': {
            'color': '#FFC533FF',
            'order': 25,
            'detector': 'fixed threshold',
            'reator': 'volatile exchange',
        },
        'D-N-V': {
            'color': '#D579FFFF',
            'order': 26,
            'detector': 'normal threshold',
            'reator': 'volatile exchange',
        },
        'D-W-V': {
            'color': '#17A2E3FF',
            'order': 27,
            'detector': 'weight threshold',
            'reator': 'volatile exchange',
        },
        'D-S-V': {
            'color': '#19FFB2CC',
            'order': 28,
            'detector': 'statistical threshold',
            'reator': 'volatile exchange',
        },
    }

    analyses = {
        'all': [
            'D-F-E', 'D-F-P', 'D-F-V', 'D-N-E', 'D-N-P', 'D-N-V', 'D-S-E',
            'D-S-P', 'D-S-V', 'D-W-E', 'D-W-P', 'D-W-V', 'F-F-E', 'F-N-E',
            'F-S-E', 'F-W-E'#, 'S-F-E', 'S-F-P', 'S-F-V', 'S-N-E', 'S-N-P',
            #'S-N-V', 'S-S-E', 'S-S-P', 'S-S-V', 'S-W-E', 'S-W-P', 'S-W-V',
        ],

        'fixed-exchange': [
            'F-F-E',
            'F-N-E',
            'F-W-E',
            'F-S-E',
        ],

#         'single-exchange': [
#             'S-F-E',
#             'S-N-E',
#             'S-W-E',
#             'S-S-E',
#         ],
#
#         'single-volatile': [
#             'S-F-V',
#             'S-N-V',
#             'S-W-V',
#             'S-S-V',
#         ],
#
#         'single-pareto': [
#             'S-F-P',
#             'S-N-P',
#             'S-W-P',
#             'S-S-P',
#         ],

        'drift-exchange': [
            'D-F-E',
            'D-N-E',
            'D-W-E',
            'D-S-E',
        ],

        'drift-volatile': [
            'D-F-V',
            'D-N-V',
            'D-W-V',
            'D-S-V',
        ],

        'drift-pareto': [
            'D-F-P',
            'D-N-P',
            'D-W-P',
            'D-S-P',
        ],

        'drift-fixed': [
            'F-F-E',
            'D-F-E',
            'D-F-V',
            'D-F-P',
        ],

        'non-weighted-pareto': [
            'F-N-E',
            'D-N-E',
            'D-N-V',
            'D-N-P',
        ],

        'weighted-volatile': [
            'F-W-E',
            'D-W-E',
            'D-W-V',
            'D-W-P',
        ],

        'statistic-volatile': [
            'F-S-E',
            'D-S-E',
            'D-S-V',
            'D-S-P',
        ],

#         'fixed-pareto': [
#             'F-F-E',
#             'S-F-E',
#             'D-F-E',
#             'S-F-P',
#             'D-F-P',
#         ],
#
#         'non-weighted-pareto': [
#             'F-N-E',
#             'S-N-E',
#             'D-N-E',
#             'S-N-P',
#             'D-N-P',
#         ],
#
#         'weighted-pareto': [
#             'F-W-E',
#             'S-W-E',
#             'D-W-E',
#             'S-W-P',
#             'D-W-P',
#         ],
#
#         'statistical-pareto': [
#             'F-S-E',
#             'S-S-E',
#             'D-S-E',
#             'S-S-P',
#             'D-S-P',
#         ],
#
#         'fixed-volatile': [
#             'F-F-E',
#             'S-F-E',
#             'D-F-E',
#             'S-F-V',
#             'D-F-V',
#         ],
#
#         'non-weighted-volatile': [
#             'F-N-E',
#             'S-N-E',
#             'D-N-E',
#             'S-N-V',
#             'D-N-V',
#         ],
#
#         'weighted-volatile': [
#             'F-W-E',
#             'S-W-E',
#             'D-W-E',
#             'S-W-V',
#             'D-W-V',
#         ],
#
#         'statistic-volatile': [
#             'F-S-E',
#             'S-S-E',
#             'D-S-E',
#             'S-S-V',
#             'D-S-V',
#         ],
    }

    datasets = get_dataset_names(args.files)

    for data in datasets:
        df = read_csv(data, sep="\t")
        data_name = data.split(".")[0].split("_")[-1]

        for analysis_name, approaches_lst in analyses.items():
            cols = select_columns(df.columns.to_list(), approaches_lst)
            generate_cd_diagram(
                data,
                list(map(lambda x: x + 1, cols)),
                approaches_lst,
                analysis_name,
                args.files,
                args.decrease,
            )


        # By order
        sort_data = dict(sorted(labels.items(), key=lambda x: x[1]['order']))
        # colors = [sort_data[i]['color'] for i in sort_data.keys()]
        # create_plot(
        #     df[sort_data.keys()],
        #     colors,
        #     f'{data_name}-order',
        #     f'{data_name}-order',
        #     args.files,
        #     True
        # )

        # # By detector
        # sort_data = dict(
        #     sorted(
        #         labels.items(),
        #         key=lambda x: (x[1]['detector'], x[1]['order'])
        #     )
        # )
        # colors = [sort_data[i]['color'] for i in sort_data.keys()]
        # create_plot(
        #     df[sort_data.keys()],
        #     colors,
        #     f'{data_name}-detector',
        #     f'{data_name}-detector',
        #     args.files,
        #     True
        # )

        # # By reactor
        # sort_data = dict(
        #     sorted(
        #         labels.items(),
        #         key=lambda x: (x[1]['reator'], x[1]['detector'], x[1]['order'])
        #     )
        # )
        # colors = [sort_data[i]['color'] for i in sort_data.keys()]
        # create_plot(
        #     df[sort_data.keys()],
        #     colors,
        #     f'{data_name}-reactor',
        #     f'{data_name}-reactor',
        #     args.files,
        #     True
        # )

#     create_plot(df, labels, cols, data_name, data_name, args.files, True)
#     generate_cd_diagrm(
#         data,
#         list(range(1, len(labels) + 1)),
#         labels,
#         data_name,
#         args.files,
#         args.decrease,
#     )
#
#     # Detector: Fixed Threshold
#     _lst = [0, 1, 4, 5]
#     print(
#         f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
#         f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
#         "\n\n",
#     )
#     create_plot(
#         df.iloc[:, _lst],
#         list(operator.itemgetter(*_lst)(labels)),
#         list(operator.itemgetter(*_lst)(cols)),
#         data_name + " FixedThreshold",
#         data_name + "_fixedthreshold",
#         args.files,
#     )
#     generate_cd_diagrm(
#         data,
#         [1, 2, 5, 6],
#         list(operator.itemgetter(*_lst)(labels)),
#         "fixedthreshold",
#         args.files,
#         args.decrease,
#     )
#
#     # Detector: Normal
#     _lst = [2, 3, 6, 7]
#     print(
#         f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
#         f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
#         "\n\n",
#     )
#     create_plot(
#         df.iloc[:, _lst],
#         list(operator.itemgetter(*_lst)(labels)),
#         list(operator.itemgetter(*_lst)(cols)),
#         data_name + " Normal",
#         data_name + "_normal",
#         args.files,
#     )
#     generate_cd_diagrm(
#         data,
#         [3, 4, 7, 8],
#         list(operator.itemgetter(*_lst)(labels)),
#         "normal",
#         args.files,
#         args.decrease,
#     )
#
#     # # Detector: Statistical
#     # _lst = list(range(12, 18))
#     # print(
#     #     f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
#     #     f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
#     #     "\n\n",
#     # )
#     # create_plot(
#     #     df.iloc[:, _lst],
#     #     list(operator.itemgetter(*_lst)(labels)),
#     #     list(operator.itemgetter(*_lst)(cols)),
#     #     data_name + " Statistical",
#     #     data_name + "_statistical",
#     #     args.files,
#     # )
#     # generate_cd_diagrm(
#     #     data,
#     #     list(range(13, 19)),
#     #     list(operator.itemgetter(*_lst)(labels)),
#     #     "statistical",
#     #     args.files,
#     #     args.decrease,
#     # )
#
#     # # Reactor: Exchange
#     # _lst = [0, 1, 6, 7, 12, 13]
#     # print(
#     #     f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
#     #     f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
#     #     "\n\n",
#     # )
#     # create_plot(
#     #     df.iloc[:, _lst],
#     #     list(operator.itemgetter(*_lst)(labels)),
#     #     list(operator.itemgetter(*_lst)(cols)),
#     #     data_name + " Exchange",
#     #     data_name + "_exchange",
#     #     args.files,
#     # )
#     # generate_cd_diagrm(
#     #     data,
#     #     [1, 2, 7, 8, 13, 14],
#     #     list(operator.itemgetter(*_lst)(labels)),
#     #     "exchange",
#     #     args.files,
#     #     args.decrease,
#     # )
#
#     # # Reactor: Pareto
#     # _lst = [2, 3, 8, 9, 14, 15]
#     # print(
#     #     f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
#     #     f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
#     #     "\n\n",
#     # )
#     # create_plot(
#     #     df.iloc[:, _lst],
#     #     list(operator.itemgetter(*_lst)(labels)),
#     #     list(operator.itemgetter(*_lst)(cols)),
#     #     data_name + " Pareto",
#     #     data_name + "_pareto",
#     #     args.files,
#     # )
#     # generate_cd_diagrm(
#     #     data,
#     #     [3, 4, 9, 10, 15, 16],
#     #     list(operator.itemgetter(*_lst)(labels)),
#     #     "pareto",
#     #     args.files,
#     #     args.decrease,
#     # )
#
#     # # Reactor: Volatile Exchange
#     # _lst = [4, 5, 10, 11, 16, 17]
#     # print(
#     #     f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
#     #     f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
#     #     "\n\n",
#     # )
#     # create_plot(
#     #     df.iloc[:, _lst],
#     #     list(operator.itemgetter(*_lst)(labels)),
#     #     list(operator.itemgetter(*_lst)(cols)),
#     #     data_name + " VolatileExchange",
#     #     data_name + "_volatile",
#     #     args.files,
#     # )
#     # generate_cd_diagrm(
#     #     data,
#     #     [5, 6, 11, 12, 17, 18],
#     #     list(operator.itemgetter(*_lst)(labels)),
#     #     "volatile",
#     #     args.files,
#     #     args.decrease,
#     # )
#
#     # Strategy: Simple
#     _lst = list(range(1, 8, 2))
#     print(
#         f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
#         f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
#         "\n\n",
#     )
#     create_plot(
#         df.iloc[:, _lst],
#         list(operator.itemgetter(*_lst)(labels)),
#         list(operator.itemgetter(*_lst)(cols)),
#         data_name + " simple",
#         data_name + "_simple",
#         args.files,
#     )
#     generate_cd_diagrm(
#         data,
#         list(range(2, 9, 2)),
#         list(operator.itemgetter(*_lst)(labels)),
#         "simple",
#         args.files,
#         args.decrease,
#     )
#
#     # Strategy: Drift
#     _lst = list(range(0, 8, 2))
#     print(
#         f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
#         f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
#         "\n\n",
#     )
#     create_plot(
#         df.iloc[:, _lst],
#         list(operator.itemgetter(*_lst)(labels)),
#         list(operator.itemgetter(*_lst)(cols)),
#         data_name + " drift",
#         data_name + "_drift",
#         args.files,
#     )
#     generate_cd_diagrm(
#         data,
#         list(range(1, 8, 2)),
#         list(operator.itemgetter(*_lst)(labels)),
#         "drift",
#         args.files,
#         args.decrease,
#     )
#
#     # aggregated = [read_csv(data, sep="\t") for data in datasets]
#     # create_plot(
#     #     aggregated,
#     #     labels,
#     #     cols,
#     #     "general",
#     #     "general",
#     #     args.files,
#     # )
#
#     # create_plot(
#     #     aggregated.iloc[:, [0, 1, 4, 5, 8, 9]],
#     #     list(operator.itemgetter(*[0, 1, 4, 5, 8, 9])(labels)),
#     #     cols2,
#     #     "general Exchange",
#     #     "general_exchange",
#     #     args.files,
#     # )
#
#     # create_plot(
#     #     aggregated.iloc[:, [2, 3, 6, 7, 10, 11]],
#     #     list(operator.itemgetter(*[2, 3, 6, 7, 10, 11])(labels)),
#     #     cols2,
#     #     "general Pareto",
#     #     "general_pareto",
#     #     args.files,
#     # )
#
#     # create_plot(
#     #     aggregated.iloc[:, [1, 3, 5, 7, 9, 11]],
#     #     list(operator.itemgetter(*[1, 3, 5, 7, 9, 11])(labels)),
#     #     cols2,
#     #     "general simple",
#     #     "general_simple",
#     #     args.files,
#     # )
#
#     # create_plot(
#     #     aggregated.iloc[:, [0, 2, 4, 6, 8, 10]],
#     #     list(operator.itemgetter(*[0, 2, 4, 6, 8, 10])(labels)),
#     #     cols2,
#     #     "general drift",
#     #     "general_drift",
#     #     args.files,
#     # )
