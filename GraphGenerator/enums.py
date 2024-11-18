from enum import Enum, auto


class Columns(Enum):
    feature1 = "feature1"
    feature2 = "feature2"
    feature3 = "feature3"
    feature4 = "feature4"
    feature5 = "feature5"
    feature6 = "feature6"
    feature7 = "feature7"
    feature8 = "feature8"
    feature9 = "feature9"
    feature10 = "feature10"
    feature11 = "feature11"
    feature12 = "feature12"
    feature13 = "feature13"
    feature14 = "feature14"
    feature15 = "feature15"
    feature16 = "feature16"
    feature17 = "feature17"
    feature18 = "feature18"
    ClaimAmount = "ClaimAmount"


class GraphType(Enum):
    SCATTER = auto()
    BAR = auto()
    LINE = auto()
    HISTO = auto()
