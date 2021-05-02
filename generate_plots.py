"""
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

The generation of visualizations.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sparseconvnet as scn
import torch
from PIL import Image
from sklearn import manifold
from torchvision.transforms import transforms

from data_provider import flip_events_along_y, random_flip_events_along_x, generate_event_histogram, \
    random_swap_channels, random_change_brightness, events_to_voxel_grid

IMAGES_DIRECTORY = "./images/"

try:
    os.mkdir(IMAGES_DIRECTORY)
except OSError:
    print("Creation of the directory %s failed" % IMAGES_DIRECTORY)
else:
    print("Successfully created the directory %s " % IMAGES_DIRECTORY)

plt.style.use('seaborn')

tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)


def set_size(width=345, fraction=1):
    # This method is taken from https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


sns.set(rc={'figure.figsize': (set_size())})


def plot_histogram(histogram, filename, id):
    x = np.interp(histogram[:, :, 1], (histogram[:, :, 1].min(), histogram[:, :, 1].max()), (128, 255))
    img = Image.fromarray(np.uint8(x), 'L')
    img.save("./images/" + filename.split('/')[-2] + id + ' pos.jpg')
    x = np.interp(histogram[:, :, 0], (histogram[:, :, 0].min(), histogram[:, :, 0].max()), (128, 0))
    img = Image.fromarray(np.uint8(x), 'L')
    img.save("./images/" + filename.split('/')[-2] + id + ' neg.jpg')


def plot_3d_events(events, id=""):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    events_ = flip_events_along_y(events)
    events_ = random_flip_events_along_x(events_, p=1)
    fig = plt.figure(figsize=set_size(800))
    ax = Axes3D(fig)
    scatter3d = ax.scatter(events_[:, 2], events_[:, 0], events_[:, 1], s=2)
    ax.set_xlabel('Time', labelpad=20, fontsize=22)
    ax.set_ylabel('Width', labelpad=20, fontsize=22)
    ax.set_zlabel('Height', labelpad=20, fontsize=22)
    ax.tick_params(labelsize=20)
    fig.savefig("./images/3d_events_{}.pdf".format(id), format="pdf", bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_voxel_grid(voxel_grid, filename):
    for i in range(0, voxel_grid.shape[0]):
        x = np.interp(voxel_grid[i, :, :], (voxel_grid[i, :, :].min(), voxel_grid[i, :, :].max()), (255, 0))
        img = Image.fromarray(np.uint8(x), 'L')
        img.save("./images/" + filename.split('/')[-2] + "_bin" + str(i) + '.jpg')


events = np.load("./data/N-Caltech256/training/harp/098_0073.npy").astype(np.float32)
plot_3d_events(events, "harp")

events = np.load("./data/N-Caltech256/training/tennis-shoes/255_0012.npy").astype(np.float32)
plot_3d_events(events, id="tennis-shoes")

# ----------------------------------------

filename = "./data/N-Caltech256/training/tomato/221_0024.npy"
events = np.load(filename).astype(np.float32)
histogram = generate_event_histogram(events, (180, 240))
plot_histogram(histogram, filename, "_white_background")

filename = "./data/N-Caltech256/training/tomato/221_0005.npy"
events = np.load(filename).astype(np.float32)
histogram = generate_event_histogram(events, (180, 240))
plot_histogram(histogram, filename, "_complex_background")

filename = "./data/N-Caltech256/training/light-house/132_0050.npy"
events = np.load(filename).astype(np.float32)
histogram = generate_event_histogram(events, (180, 240))

side = (180, 240)
rnd_rot = transforms.RandomRotation(10., interpolation=transforms.InterpolationMode.BILINEAR)
rnd_hflip = transforms.RandomHorizontalFlip(p=1)
rnd_erase = transforms.RandomErasing(p=1, scale=(0.02, 0.1))
rnd_blur = transforms.GaussianBlur((int(side[1] * .1) if int(side[1] * .1) % 2 != 0 else int(side[1] * .1) + 1),
                                   sigma=(1., 1.4))
rnd_resizedcrop = transforms.RandomResizedCrop(size=side, scale=(0.6, 0.7),
                                               ratio=(0.75, 1.3333333333333333),
                                               interpolation=transforms.InterpolationMode.BILINEAR)
transform = transforms.Compose([rnd_resizedcrop, rnd_hflip, rnd_blur, rnd_rot, rnd_erase])

plot_histogram(histogram, filename, "original")
for i, tr in enumerate(transform.transforms):
    histogram_swaped_axis = np.moveaxis(histogram.copy(), -1, 0)
    histogram_transformed = tr(torch.from_numpy(histogram_swaped_axis)).numpy()
    histogram_transformed = np.moveaxis(histogram_transformed, 0, -1)
    plot_histogram(histogram_transformed, filename, str(i))
plot_histogram(random_swap_channels(histogram, p=1), filename, "swap")
plot_histogram(random_change_brightness(histogram, p=1), filename, "change_brightness")

# ----------------------------------------

filename = "./data/N-Caltech256/training/tennis-shoes/255_0012.npy"
events = np.load(filename).astype(np.float32)
histogram = generate_event_histogram(events, (180, 240))
plot_histogram(histogram, filename, "")

for i, tr in enumerate(transform.transforms):
    histogram_swaped_axis = np.moveaxis(histogram.copy(), -1, 0)
    histogram_transformed = tr(torch.from_numpy(histogram_swaped_axis)).numpy()
    histogram_transformed = np.moveaxis(histogram_transformed, 0, -1)
    plot_histogram(histogram_transformed, filename, str(i))

# --------------------------------------------

filename = "./data/N-Caltech256/training/owl/152_0030.npy"
events = np.load(filename).astype(np.float32)
voxel_grid = events_to_voxel_grid(events, 10, 240, 180)
plot_voxel_grid(voxel_grid, filename)


# --------------------------------------------


def plot_loss(filename1, line_name1,
              x_label, y_label, title, hue_name,
              filename2=None, line_name2=None):
    data1 = pd.read_csv(filename1)
    data1[hue_name] = line_name1
    epochs = data1[["epoch"]].to_numpy().flatten() + 1
    if filename2 is not None:
        data2 = pd.read_csv(filename2)
        data2[hue_name] = line_name2
        data = data1.append(data2)
        data["epoch"] = data["epoch"] + 1
    else:
        data = data1
        data["epoch"] = data["epoch"] + 1
    sns.set(font_scale=1.5)
    g = sns.lineplot(data=data, x="epoch", y="loss", hue=hue_name, style=hue_name, markers=True, dashes=False)
    g.set(xticks=np.insert(epochs[0::4], len(epochs[0::4]) - 1, len(epochs)))
    g.set_ylabel(y_label)
    g.set_xlabel(x_label)
    g.set_title(title)
    g.figure.savefig("./images/loss_plstm.pdf", format="pdf", bbox_inches="tight")
    plt.close()


plot_loss(
    filename1="./checkpoint/standard/ncaltech101/plstm/setup1/ncaltech101/log_standard_ncaltech101_plstm_seed_10_batch_30.cvs",
    line_name1="1\%",
    filename2="./checkpoint/standard/ncaltech101/plstm/setup2/ncaltech101/log_standard_ncaltech101_plstm_seed_10_batch_30.cvs",
    line_name2="5\%",
    hue_name="Pct. of events",
    x_label="Epoch",
    y_label="Loss score",
    title="Phased LSTM: N-Caltech101"
)


# --------------------------------------------

def plot_accuracy():
    # PLSTM
    d = {"Accuracy": [34.13, 35.35, 30.09, 30.90],
         "Dataset": ["Train", "Train", "Test", "Test"],
         "Pct. of events": ["1\%", "5\%",
                            "1\%", "5\%"]}
    df = pd.DataFrame(data=d)
    sns.set(font_scale=1.5)
    g = sns.catplot(x="Dataset", y="Accuracy", hue="Pct. of events", data=df, kind="bar",
                    height=4, aspect=.8)
    g.axes[0, 0].set_ylabel("Accuracy")
    g.axes[0, 0].set_xlabel("Dataset")
    g.axes[0, 0].set_title("Phased LSTM: N-Caltech101")
    plt.subplots_adjust(top=0.9, bottom=0.15, right=0.6, left=0.1)
    g.savefig("./images/accuracy_plstm.pdf", format="pdf", bbox_inches="tight")
    plt.close()


plot_accuracy()

# --------------------------------------------
data = np.load("./data/ncaltech12_ssl_features_test_data_histogram.npy")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
data_tsne = tsne.fit_transform(data)
sns.set(font_scale=.5)
g = sns.scatterplot(
    x=data_tsne[:, 0],
    y=data_tsne[:, 1],
    hue=data[:, -1],
    palette=sns.color_palette("bright", len(np.unique(data[:, -1]))),
    legend=False,
    style=data[:, -1],
    alpha=0.5
)

# return the figure
g.figure.savefig(
    "./images/ncaltech12_ssl_features_test_data_histogram_tsne_classes-{}.pdf".format(len(np.unique(data[:, -1]))),
    format="pdf",
    bbox_inches="tight")
plt.close()

data = np.load("./data/ncaltech12_ssl_features_test_data_voxel.npy")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
data_tsne = tsne.fit_transform(data)
sns.set(font_scale=.5)
g = sns.scatterplot(
    x=data_tsne[:, 0],
    y=data_tsne[:, 1],
    hue=data[:, -1],
    palette=sns.color_palette("bright", len(np.unique(data[:, -1]))),
    legend=False,
    style=data[:, -1],
    alpha=0.5
)

g.figure.savefig(
    "./images/ncaltech12_ssl_features_test_data_voxel_tsne_classes-{}.pdf".format(len(np.unique(data[:, -1]))),
    format="pdf",
    bbox_inches="tight")
plt.close()

# --------------------------------------------

histogram = pd.read_csv("./checkpoint/simclr/ncaltech12/Histogram/log_simclr_ncaltech12_scnn_seed_10_batch_64.cvs")
histogram.rename(columns={"loss": "histogram"}, inplace=True)
histogram.drop(['score'], axis=1, inplace=True)

voxel = pd.read_csv("./checkpoint/simclr/ncaltech12/Voxel/log_simclr_ncaltech12_scnn_seed_10_batch_64.cvs")
voxel.rename(columns={"loss": "voxel grid"}, inplace=True)
voxel.drop(['score'], axis=1, inplace=True)

df = histogram.merge(voxel)
df = df.melt('epoch', value_vars=['histogram', 'voxel grid'], value_name='Loss')
df["epoch"] = df["epoch"] + 1
df.rename(columns={"variable": "Event representation", "epoch": "Epoch"}, inplace=True)
sns.set(font_scale=1.2)
g = sns.lineplot(x="Epoch", y="Loss", hue='Event representation', data=df,
                 style='Event representation')
x_indices = [i * 50 for i in range(1, 9)]
x_indices.insert(0, 1)
g.set(xticks=x_indices)
g.set_title("Self-supervised training on N-Caltech256-12")
g.figure.savefig("./images/ncaltech12_ssl_train_loss.pdf",
                 format="pdf",
                 bbox_inches="tight")
plt.close()


def dense_to_sparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
    locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

    select_indices = non_zero_indices.split(1, dim=1)
    features = torch.squeeze(dense_tensor[select_indices], dim=-2)

    return locations, features


def create_input(input_shape):
    i = np.zeros(input_shape)
    np.fill_diagonal(i, 1)
    d = np.uint8(np.interp(i, (i.min(), i.max()), (128, 255)))
    img = Image.fromarray(np.uint8(d), 'L')
    img.save("./images/input.jpg")
    return torch.from_numpy(i)


def apply_dense_conv(input_, filename):
    m = torch.nn.Conv2d(1, 1, 3, stride=1)
    m.bias.data.fill_(0.0)
    m.weight.data.fill_(1 / 3)
    out = m(input_.float())
    d = out.squeeze().detach().numpy()
    d = np.uint8(np.interp(d, (d.min(), d.max()), (128, 255)))
    img = Image.fromarray(np.uint8(d), 'L')
    img.save(filename)
    print("output of shape {} for {}".format(out.shape, filename))
    return out


def apply_submanifold_conv(input_, filename, submanifold_output):
    sm = scn.SubmanifoldConvolution(2, 1, 1, 3, False)
    sm.weight.data.fill_(1 / 3)
    cnn_spatial_output_size = [submanifold_output[0], submanifold_output[1]]
    spatial_size = sm.input_spatial_size(torch.LongTensor(cnn_spatial_output_size))
    input_layer = scn.InputLayer(dimension=2, spatial_size=spatial_size, mode=2)
    locations, features = dense_to_sparse(input_)
    x = [locations, features.float(), input_.shape[0]]
    input_ = input_layer(x)
    out = sm(input_)
    sd = scn.SparseToDense(2, 1)
    d = sd(out)
    d_ = d.squeeze().detach()
    d = np.uint8(np.interp(d_, (d_.min(), d_.max()), (128, 255)))
    img = Image.fromarray(np.uint8(d), 'L')
    img.save(filename)
    print("output of shape {} for {}".format(d_.shape, filename))
    return d_


input_shape = (128, 128)
i = create_input(input_shape)
out = apply_dense_conv(i.view(1, 1, i.shape[0], i.shape[1]), "./images/conv1.jpg")
for k in range(2, 20):
    out = apply_dense_conv(out, "./images/conv{}.jpg".format(k))

x = input_shape[0] - 2
y = input_shape[1] - 2
out = apply_submanifold_conv(i.view(i.shape[0], i.shape[1], 1), "./images/sconv1.jpg", [x, y])
for k in range(2, 20):
    x -= 2
    y -= 2
    out = apply_submanifold_conv(out.view(out.shape[0], out.shape[1], 1), "./images/sconv{}.jpg".format(k), [x, y])
