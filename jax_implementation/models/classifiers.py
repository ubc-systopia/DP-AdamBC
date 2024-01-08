import configlib
from models.cnn import CNN2, CNN5
from models.gn_resnet import GNResNet18
from models.wide_resnet import WideResNet

parser = configlib.add_parser("Classifier config")
parser.add_argument("--classifier_model", default="CNN2", type=str, metavar="MODEL_NAME",
        help="The type of classifier to train (CNN2, CNN5, ResNet18).")
parser.add_argument("--num_classes", default=10, type=int,
        help="The number of classes in the classifier's output.")

parser.add_argument("--activation", default="leaky_relu", type=str, metavar="ACTIVATION_NAME",
        help="The type of classifier to train (relu, leaky_relu, tanh, elu).")
parser.add_argument("--negative_slope", default=.2, type=float,
        help="The negative slope when using leaky_relu.")
parser.add_argument("--elu_alpha", default=1., type=float,
        help="The alpha param of ELU activations (when ELU is used).")
parser.add_argument("--normalization", default="none", type=str,
                    help="Choices are 'none', 'group_norm'")
parser.add_argument("--weight_standardization", default=False, action='store_true')

# Wide-ResNet
parser.add_argument("--resnet_width", default=4, type=int)
parser.add_argument("--resnet_depth", default=16, type=int)


def get_classifier(conf: configlib.Config):
    Model = globals()[conf.classifier_model]  # get the hk.Module for the model from class name
    def model_fn(*args, **kwargs):
        return Model(conf)(*args, **kwargs)
    return model_fn

