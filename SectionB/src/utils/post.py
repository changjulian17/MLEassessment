from pydantic import BaseModel
from typing import List


class UserTrack(BaseModel):
    """
        Represents a single music track.

        Attributes:
            trackID (int): The unique identifier of the track.
            title (str): The title of the track.
            tags (str): Tags associated with the track.
            loudness (float): Loudness of the track.
            tempo (float): Tempo of the track.
            time_signature (float): Time signature of the track.
            key (float): Key of the track.
            mode (float): Mode of the track.
            duration (float): Duration of the track.
            vect_1 to vect_148 (float): Features of the track.

    """
    trackID: int
    title: str
    tags: str
    loudness: float
    tempo: float
    time_signature: float
    key: float
    mode: float
    duration: float
    vect_1: float
    vect_2: float
    vect_3: float
    vect_4: float
    vect_5: float
    vect_6: float
    vect_7: float
    vect_8: float
    vect_9: float
    vect_10: float
    vect_11: float
    vect_12: float
    vect_13: float
    vect_14: float
    vect_15: float
    vect_16: float
    vect_17: float
    vect_18: float
    vect_19: float
    vect_20: float
    vect_21: float
    vect_22: float
    vect_23: float
    vect_24: float
    vect_25: float
    vect_26: float
    vect_27: float
    vect_28: float
    vect_29: float
    vect_30: float
    vect_31: float
    vect_32: float
    vect_33: float
    vect_34: float
    vect_35: float
    vect_36: float
    vect_37: float
    vect_38: float
    vect_39: float
    vect_40: float
    vect_41: float
    vect_42: float
    vect_43: float
    vect_44: float
    vect_45: float
    vect_46: float
    vect_47: float
    vect_48: float
    vect_49: float
    vect_50: float
    vect_51: float
    vect_52: float
    vect_53: float
    vect_54: float
    vect_55: float
    vect_56: float
    vect_57: float
    vect_58: float
    vect_59: float
    vect_60: float
    vect_61: float
    vect_62: float
    vect_63: float
    vect_64: float
    vect_65: float
    vect_66: float
    vect_67: float
    vect_68: float
    vect_69: float
    vect_70: float
    vect_71: float
    vect_72: float
    vect_73: float
    vect_74: float
    vect_75: float
    vect_76: float
    vect_77: float
    vect_78: float
    vect_79: float
    vect_80: float
    vect_81: float
    vect_82: float
    vect_83: float
    vect_84: float
    vect_85: float
    vect_86: float
    vect_87: float
    vect_88: float
    vect_89: float
    vect_90: float
    vect_91: float
    vect_92: float
    vect_93: float
    vect_94: float
    vect_95: float
    vect_96: float
    vect_97: float
    vect_98: float
    vect_99: float
    vect_100: float
    vect_101: float
    vect_102: float
    vect_103: float
    vect_104: float
    vect_105: float
    vect_106: float
    vect_107: float
    vect_108: float
    vect_109: float
    vect_110: float
    vect_111: float
    vect_112: float
    vect_113: float
    vect_114: float
    vect_115: float
    vect_116: float
    vect_117: float
    vect_118: float
    vect_119: float
    vect_120: float
    vect_121: float
    vect_122: float
    vect_123: float
    vect_124: float
    vect_125: float
    vect_126: float
    vect_127: float
    vect_128: float
    vect_129: float
    vect_130: float
    vect_131: float
    vect_132: float
    vect_133: float
    vect_134: float
    vect_135: float
    vect_136: float
    vect_137: float
    vect_138: float
    vect_139: float
    vect_140: float
    vect_141: float
    vect_142: float
    vect_143: float
    vect_144: float
    vect_145: float
    vect_146: float
    vect_147: float
    vect_148: float


class UserTracks(BaseModel):
    """
        Represents a list of music tracks.

        Attributes:
            tracks (List[UserTrack]): List of UserTrack objects representing individual music tracks.

    """
    tracks: List[UserTrack]
