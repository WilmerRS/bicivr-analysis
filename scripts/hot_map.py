import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image


def convert_to_xy(x, y):
    s_x, s_y = 1.2406875598285265, -1.2287975162804792
    t_x, t_y = -689.320501935323, 1437.5880130243834

    return s_x * x + t_x, s_y * y + t_y


def _open_image(image_url):
    return Image.open(image_url)


def create_figure(image_url):
    fig, ax = plt.subplots()
    ax.imshow(_open_image(image_url))
    return fig, ax


def display_figure(fig, ax):
    plt.show()


def draw_circle(ax, x, y, radius=5, color_hex='#0000ff'):
    color_rgb = mcolors.hex2color(color_hex)
    color_rgba = color_rgb + (0.3,)

    circle = patches.Circle(
        (x, y),
        radius=radius,
        edgecolor=color_rgb,
        facecolor=color_rgba,
        lw=1
    )
    ax.add_patch(circle)
    ax.set_aspect('equal')


def draw_arrow(ax: plt.Axes, points, color_hex='#0000ff', stroke_width=1):
    color_rgb = mcolors.hex2color(color_hex)
    current_point = points[0]

    is_last = False

    for point in points:
        is_last = point == points[-1]
        ax.arrow(
            x=current_point[0],
            y=current_point[1],
            dx=point[0] - current_point[0],
            dy=point[1] - current_point[1],
            head_width=4 if is_last else 0,
            head_length=4 if is_last else 0,
            fc=color_rgb,
            ec=color_rgb
        )
        current_point = point

    ax.set_aspect('equal')


def main():
    # games = pd.read_csv('games.csv')
    games = pd.read_csv('position_games.csv')

    game_map_image_url = 'map.png'

    # example 1 = (1077, 800) -> (646.9, 454.55)
    # example 2 = (1317.27, 1064.12) -> (945, 130)

    x_A, y_A = 1077, 800
    x_B, y_B = convert_to_xy(x_A, y_A)

    print(x_B, y_B)

    fig, ax = create_figure(game_map_image_url)
    # display_figure(fig, ax)

    # draw_circle(ax, x_B, y_B)
    # display_figure(fig, ax)

    games['vehicle_position_mean'] = list(zip(
        games['vehicle_position_x_mean'],
        games['vehicle_position_z_mean']
    ))
    games['vehicle_position_mean'] = games['vehicle_position_mean'].apply(
        lambda x: convert_to_xy(x[0], x[1])
    )

    # bikes

    games['bike_position_mean'] = list(zip(
        games['bike_position_x_mean'],
        games['bike_position_z_mean']
    ))
    games['bike_position_mean'] = games['bike_position_mean'].apply(
        lambda x: convert_to_xy(x[0], x[1])
    )
    games['bike_position_min'] = list(zip(
        games['bike_position_x_min'],
        games['bike_position_z_min']
    ))
    games['bike_position_min'] = games['bike_position_min'].apply(
        lambda x: convert_to_xy(x[0], x[1])
    )
    games['bike_position_max'] = list(zip(
        games['bike_position_x_max'],
        games['bike_position_z_max']
    ))
    games['bike_position_max'] = games['bike_position_max'].apply(
        lambda x: convert_to_xy(x[0], x[1])
    )

    # vehicles

    games['vehicle_position_mean'] = list(zip(
        games['vehicle_position_x_mean'],
        games['vehicle_position_z_mean']
    ))
    games['vehicle_position_mean'] = games['vehicle_position_mean'].apply(
        lambda x: convert_to_xy(x[0], x[1])
    )
    games['vehicle_position_min'] = list(zip(
        games['vehicle_position_x_min'],
        games['vehicle_position_z_min']
    ))
    games['vehicle_position_min'] = games['vehicle_position_min'].apply(
        lambda x: convert_to_xy(x[0], x[1])
    )
    games['vehicle_position_max'] = list(zip(
        games['vehicle_position_x_max'],
        games['vehicle_position_z_max']
    ))
    games['vehicle_position_max'] = games['vehicle_position_max'].apply(
        lambda x: convert_to_xy(x[0], x[1])
    )

    games = games[games['real_risk'] == 'collision']
    # games = games[games['real_risk'] != 'warning']

    colors = {
        'collision': '#ba0606',
        'danger': '#e0c91b',
        'warning': '#1be050',
    }

    # for game in games.itertuples():
    #     # draw_circle(
    #     #     ax,
    #     #     game.vehicle_position_mean[0],
    #     #     game.vehicle_position_mean[1],
    #     #     color_hex='#ff0000'
    #     # )
    #     draw_circle(
    #         ax,
    #         game.bike_position_min[0],
    #         game.bike_position_min[1],
    #         radius=2,
    #         color_hex="#e0c91b"
    #     )
    #     draw_circle(
    #         ax,
    #         game.bike_position_mean[0],
    #         game.bike_position_mean[1],
    #         radius=2,
    #         color_hex="#1be050"
    #     )
    #     draw_circle(
    #         ax,
    #         game.bike_position_max[0],
    #         game.bike_position_max[1],
    #         radius=2,
    #         color_hex="#278a85"
    #     )

    for game in games.itertuples():
        draw_arrow(
            ax,
            [game.bike_position_min, game.bike_position_mean, game.bike_position_max],
            color_hex="#ba0606"
        )
        # draw_arrow(
        #     ax,
        #     [game.vehicle_position_min, game.vehicle_position_mean, game.vehicle_position_max],
        #     color_hex="#3a32a8"
        # )

    display_figure(fig, ax)


if __name__ == '__main__':
    main()
