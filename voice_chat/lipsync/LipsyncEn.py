"""
This class is a library ported from the lipsync js class created by https://github.com/met4citizen (modules/lipsync-en.mjs)

Lipsync library for english language based on the rule set contained in https://apps.dtic.mil/sti/pdfs/ADA021929.pdf
"""

import re
from attrs import define, Factory, field
from timeit_decorator import timeit
import logging
from typing import Dict, Tuple, List, Optional


@define
class LipsyncEn:
    rules = field(factory=dict, init=False)
    ops = field(factory=dict, init=False)
    viseme_durations = field(factory=dict, init=False)
    special_durations = field(factory=dict, init=False)
    digits = field(factory=list, init=False)
    ones = field(factory=list, init=False)
    tens = field(factory=list, init=False)
    teens = field(factory=list, init=False)
    symbols = field(factory=dict, init=False)
    symbols_reg = field(init=False, default=None)
    azure_to_occulus_viseme_map = field(factory=list, init=False)

    def __attrs_post_init__(self):
        # English words to Oculus visemes, adapted from the NRL Report 7948
        self.rules = {
            "A": [
                "[A] =aa",
                " [ARE] =aa RR",
                " [AR]O=aa RR",
                "[AR]#=E RR",
                " ^[AS]#=E SS",
                "[A]WA=aa",
                "[AW]=aa",
                " :[ANY]=E nn I",
                "[A]^+#=E",
                "#:[ALLY]=aa nn I",
                " [AL]#=aa nn",
                "[AGAIN]=aa kk E nn",
                "#:[AG]E=I kk",
                "[A]^+:#=aa",
                ":[A]^+ =E",
                "[A]^%=E",
                " [ARR]=aa RR",
                "[ARR]=aa RR",
                " :[AR] =aa RR",
                "[AR] =E",
                "[AR]=aa RR",
                "[AIR]=E RR",
                "[AI]=E",
                "[AY]=E",
                "[AU]=aa",
                "#:[AL] =aa nn",
                "#:[ALS] =aa nn SS",
                "[ALK]=aa kk",
                "[AL]^=aa nn",
                " :[ABLE]=E PP aa nn",
                "[ABLE]=aa PP aa nn",
                "[ANG]+=E nn kk",
                "[A]=aa",
            ],
            "B": [
                " [BE]^#=PP I",
                "[BEING]=PP I I nn",
                " [BOTH] =PP O TH",
                " [BUS]#=PP I SS",
                "[BUIL]=PP I nn",
                "[B]=PP",
            ],
            "C": [
                " [CH]^=kk",
                "^E[CH]=kk",
                "[CH]=CH",
                " S[CI]#=SS aa",
                "[CI]A=SS",
                "[CI]O=SS",
                "[CI]EN=SS",
                "[C]+=SS",
                "[CK]=kk",
                "[COM]%=kk aa PP",
                "[C]=kk",
            ],
            "D": [
                "#:[DED] =DD I DD",
                ".E[D] =DD",
                "#^:E[D] =DD",
                " [DE]^#=DD I",
                " [DO] =DD U",
                " [DOES]=DD aa SS",
                " [DOING]=DD U I nn",
                " [DOW]=DD aa",
                "[DU]A=kk U",
                "[D]=DD",
            ],
            "E": [
                "#:[E] =",
                "'^:[E] =",
                " :[E] =I",
                "#[ED] =DD",
                "#:[E]D =",
                "[EV]ER=E FF",
                "[E]^%=I",
                "[ERI]#=I RR I",
                "[ERI]=E RR I",
                "#:[ER]#=E",
                "[ER]#=E RR",
                "[ER]=E",
                " [EVEN]=I FF E nn",
                "#:[E]W=",
                "@[EW]=U",
                "[EW]=I U",
                "[E]O=I",
                "#:&[ES] =I SS",
                "#:[E]S =",
                "#:[ELY] =nn I",
                "#:[EMENT]=PP E nn DD",
                "[EFUL]=FF U nn",
                "[EE]=I",
                "[EARN]=E nn",
                " [EAR]^=E",
                "[EAD]=E DD",
                "#:[EA] =I aa",
                "[EA]SU=E",
                "[EA]=I",
                "[EIGH]=E",
                "[EI]=I",
                " [EYE]=aa",
                "[EY]=I",
                "[EU]=I U",
                "[E]=E",
            ],
            "F": ["[FUL]=FF U nn", "[F]=FF"],
            "G": [
                "[GIV]=kk I FF",
                " [G]I^=kk",
                "[GE]T=kk E",
                "SU[GGES]=kk kk E SS",
                "[GG]=kk",
                " B#[G]=kk",
                "[G]+=kk",
                "[GREAT]=kk RR E DD",
                "#[GH]=",
                "[G]=kk",
            ],
            "H": [
                " [HAV]=I aa FF",
                " [HERE]=I I RR",
                " [HOUR]=aa EE",
                "[HOW]=I aa",
                "[H]#=I",
                "[H]=H",
            ],
            "I": [
                " [IN]=I nn",
                " [I] =aa",
                "[IN]D=aa nn",
                "[IER]=I E",
                "#:R[IED] =I DD",
                "[IED] =aa DD",
                "[IEN]=I E nn",
                "[IE]T=aa E",
                " :[I]%=aa",
                "[I]%=I",
                "[IE]=I",
                "[I]^+:#=I",
                "[IR]#=aa RR",
                "[IZ]%=aa SS",
                "[IS]%=aa SS",
                "[I]D%=aa",
                "+^[I]^+=I",
                "[I]T%=aa",
                "#^:[I]^+=I",
                "[I]^+=aa",
                "[IR]=E",
                "[IGH]=aa",
                "[ILD]=aa nn DD",
                "[IGN] =aa nn",
                "[IGN]^=aa nn",
                "[IGN]%=aa nn",
                "[IQUE]=I kk",
                "[I]=I",
            ],
            "J": ["[J]=kk"],
            "K": [" [K]N=", "[K]=kk"],
            "L": ["[LO]C#=nn O", "L[L]=", "#^:[L]%=aa nn", "[LEAD]=nn I DD", "[L]=nn"],
            "M": ["[MOV]=PP U FF", "[M]=PP"],
            "N": [
                "E[NG]+=nn kk",
                "[NG]R=nn kk",
                "[NG]#=nn kk",
                "[NGL]%=nn kk aa nn",
                "[NG]=nn",
                "[NK]=nn kk",
                " [NOW] =nn aa",
                "[N]=nn",
            ],
            "O": [
                "[OF] =aa FF",
                "[OROUGH]=E O",
                "#:[OR] =E",
                "#:[ORS] =E SS",
                "[OR]=aa RR",
                " [ONE]=FF aa nn",
                "[OW]=O",
                " [OVER]=O FF E",
                "[OV]=aa FF",
                "[O]^%=O",
                "[O]^EN=O",
                "[O]^I#=O",
                "[OL]D=O nn",
                "[OUGHT]=aa DD",
                "[OUGH]=aa FF",
                " [OU]=aa",
                "H[OU]S#=aa",
                "[OUS]=aa SS",
                "[OUR]=aa RR",
                "[OULD]=U DD",
                "^[OU]^L=aa",
                "[OUP]=U OO",
                "[OU]=aa",
                "[OY]=O",
                "[OING]=O I nn",
                "[OI]=O",
                "[OOR]=aa RR",
                "[OOK]=U kk",
                "[OOD]=U DD",
                "[OO]=U",
                "[O]E=O",
                "[O] =O",
                "[OA]=O",
                " [ONLY]=O nn nn I",
                " [ONCE]=FF aa nn SS",
                "[ON'T]=O nn DD",
                "C[O]N=aa",
                "[O]NG=aa",
                " ^:[O]N=aa",
                "I[ON]=aa nn",
                "#:[ON] =aa nn",
                "#^[ON]=aa nn",
                "[O]ST =O",
                "[OF]^=aa FF",
                "[OTHER]=aa TH E",
                "[OSS] =aa SS",
                "#^:[OM]=aa PP",
                "[O]=aa",
            ],
            "P": [
                "[PH]=FF",
                "[PEOP]=PP I PP",
                "[POW]=PP aa",
                "[PUT] =PP U DD",
                "[P]=PP",
            ],
            "Q": ["[QUAR]=kk FF aa RR", "[QU]=kk FF", "[Q]=kk"],
            "R": [" [RE]^#=RR I", "[R]=RR"],
            "S": [
                "[SH]=SS",
                "#[SION]=SS aa nn",
                "[SOME]=SS aa PP",
                "#[SUR]#=SS E",
                "[SUR]#=SS E",
                "#[SU]#=SS U",
                "#[SSU]#=SS U",
                "#[SED] =SS DD",
                "#[S]#=SS",
                "[SAID]=SS E DD",
                "^[SION]=SS aa nn",
                "[S]S=",
                ".[S] =SS",
                "#:.E[S] =SS",
                "#^:##[S] =SS",
                "#^:#[S] =SS",
                "U[S] =SS",
                " :#[S] =SS",
                " [SCH]=SS kk",
                "[S]C+=",
                "#[SM]=SS PP",
                "#[SN]'=SS aa nn",
                "[S]=SS",
            ],
            "T": [
                " [THE] =TH aa",
                "[TO] =DD U",
                "[THAT] =TH aa DD",
                " [THIS] =TH I SS",
                " [THEY]=TH E",
                " [THERE]=TH E RR",
                "[THER]=TH E",
                "[THEIR]=TH E RR",
                " [THAN] =TH aa nn",
                " [THEM] =TH E PP",
                "[THESE] =TH I SS",
                " [THEN]=TH E nn",
                "[THROUGH]=TH RR U",
                "[THOSE]=TH O SS",
                "[THOUGH] =TH O",
                " [THUS]=TH aa SS",
                "[TH]=TH",
                "#:[TED] =DD I DD",
                "S[TI]#N=CH",
                "[TI]O=SS",
                "[TI]A=SS",
                "[TIEN]=SS aa nn",
                "[TUR]#=CH E",
                "[TU]A=CH U",
                " [TWO]=DD U",
                "[T]=DD",
            ],
            "U": [
                " [UN]I=I U nn",
                " [UN]=aa nn",
                " [UPON]=aa PP aa nn",
                "@[UR]#=U RR",
                "[UR]#=I U RR",
                "[UR]=E",
                "[U]^ =aa",
                "[U]^^=aa",
                "[UY]=aa",
                " G[U]#=",
                "G[U]%=",
                "G[U]#=FF",
                "#N[U]=I U",
                "@[U]=I",
                "[U]=I U",
            ],
            "V": ["[VIEW]=FF I U", "[V]=FF"],
            "W": [
                " [WERE]=FF E",
                "[WA]S=FF aa",
                "[WA]T=FF aa",
                "[WHERE]=FF E RR",
                "[WHAT]=FF aa DD",
                "[WHOL]=I O nn",
                "[WHO]=I U",
                "[WH]=FF",
                "[WAR]=FF aa RR",
                "[WOR]^=FF E",
                "[WR]=RR",
                "[W]=FF",
            ],
            "X": [" [X]=SS", "[X]=kk SS"],
            "Y": [
                "[YOUNG]=I aa nn",
                " [YOU]=I U",
                " [YES]=I E SS",
                " [Y]=I",
                "#^:[Y] =I",
                "#^:[Y]I=I",
                " :[Y] =aa",
                " :[Y]#=aa",
                " :[Y]^+:#=I",
                " :[Y]^#=I",
                "[Y]=I",
            ],
            "Z": ["[Z]=SS"],
        }

        self.ops = {
            "#": "[AEIOUY]+",
            ".": "[BDVGJLMNRWZ]",
            "%": "(?:ER|E|ES|ED|ING|ELY)",
            "&": "(?:[SCGZXJ]|CH|SH)",
            "@": "(?:[TSRDLZNJ]|TH|CH|SH)",
            "^": "[BCDFGHJKLMNPQRSTVWXZ]",
            "+": "[EIY]",
            ":": "[BCDFGHJKLMNPQRSTVWXZ]*",
            " ": "\\b",
        }

        self.viseme_durations = {
            "aa": 0.95,
            "E": 0.90,
            "I": 0.92,
            "O": 0.96,
            "U": 0.95,
            "PP": 1.08,
            "SS": 1.23,
            "TH": 1,
            "DD": 1.05,
            "FF": 1.00,
            "kk": 1.21,
            "nn": 0.88,
            "RR": 0.88,
            "DD": 1.05,
            "sil": 1,
        }
        self.viseme_durations_elevenlabs = {
            "aa": 0.99,
            "E": 1.12,
            "I": 1.01,
            "O": 1.22,
            "U": 0.92,
            "PP": 0.86,
            "SS": 0.99,
            "TH": 1.08,
            "DD": 0.95,
            "FF": 1.08,
            "kk": 0.88,
            "nn": 1.14,
            "RR": 1.02,
            "DD": 0.95,
            "sil": 1,
        }

        self.special_durations = {" ": 1, ",": 2, "-": 0.5}

        self.digits = [
            "oh",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        self.ones = [
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        self.tens = [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]
        self.teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]

        self.symbols = {
            "%": "percent",
            "€": "euros",
            "&": "and",
            "+": "plus",
            "$": "dollars",
        }
        self.symbols_reg = re.compile(r"[%€&\+\$]")

        # Do the conversion to regex.
        self.convert_rules_to_regex()

        self.azure_to_occulus_viseme_map = [
            "sil",
            "aa",
            "aa",
            "O",
            "E",
            "I",
            "I",
            "U",
            "O",
            "aa",
            "O",
            "aa",
            "RR",
            "RR",
            "nn",
            "SS",
            "CH",
            "TH",
            "FF",
            "DD",
            "kk",
            "PP",
        ]

    # Convert rules to regex
    def convert_rules_to_regex(self):
        for key in self.rules.keys():
            self.rules[key] = [
                self.convert_rule(rule, self.ops) for rule in self.rules[key]
            ]

    def convert_rule(self, rule, ops):
        pos_l = rule.index("[")
        pos_r = rule.index("]")
        pos_e = rule.index("=")
        str_left = rule[:pos_l]
        str_letters = rule[pos_l + 1 : pos_r]
        str_right = rule[pos_r + 1 : pos_e]
        str_visemes = rule[pos_e + 1 :]

        o = {"regex": "", "move": 0, "visemes": []}

        exp = "".join(ops.get(x, x) for x in str_left)
        ctx_letters = list(str_letters)
        ctx_letters[0] = ctx_letters[0].lower()
        exp += "".join(ctx_letters)
        o["move"] = len(ctx_letters)
        exp += "".join(ops.get(x, x) for x in str_right)
        o["regex"] = re.compile(exp)

        if str_visemes:
            o["visemes"] = str_visemes.split(" ")

        return o

    def convert_digit_by_digit(self, num):
        num = str(num)
        return " ".join([self.digits[int(d)] for d in num])

    def convert_sets_of_two(self, num):
        num_str = str(num)
        return f"{self.convert_tens(num_str[:2])} {self.convert_tens(num_str[2:])}"

    def convert_millions(self, num):
        if num >= 1000000:
            return f"{self.convert_millions(num // 1000000)} million {self.convert_thousands(num % 1000000)}"
        else:
            return self.convert_thousands(num)

    def convert_thousands(self, num):
        if num >= 1000:
            return f"{self.convert_hundreds(num // 1000)} thousand {self.convert_hundreds(num % 1000)}"
        else:
            return self.convert_hundreds(num)

    def convert_hundreds(self, num):
        if num > 99:
            return f"{self.ones[num // 100]} hundred {self.convert_tens(num % 100)}"
        else:
            return self.convert_tens(num)

    def convert_tens(self, num):
        num = int(num)
        if num < 10:
            return self.ones[num]
        elif 10 <= num < 20:
            return self.teens[num - 10]
        else:
            return f"{self.tens[num // 10]} {self.ones[num % 10]}"

    def convert_number_to_words(self, num):
        if num == 0:
            return "zero"
        elif (100 <= num < 1000) or (10000 <= num < 1000000):
            return self.convert_digit_by_digit(num)
        elif (1000 < num < 2000) or (2009 < num < 3000):
            return self.convert_sets_of_two(num)
        else:
            return self.convert_millions(num)

    def pre_process_text(self, s):
        s = re.sub(r'[#_*\'":;]', "", s)
        s = self.symbols_reg.sub(lambda m: f" {self.symbols[m.group()]} ", s)
        s = re.sub(r"(\d),(\d)", r"\1 point \2", s)  # Number separator
        s = re.sub(
            r"\d+", lambda m: self.convert_number_to_words(int(m.group())), s
        )  # Numbers to words
        s = re.sub(r"(\D)\1\1+", r"\1\1", s)  # max 2 repeating chars
        s = s.replace("  ", " ")  # Only one repeating space
        s = re.sub(r"[\u0300-\u036f]", "", s)  # Remove non-English diacritics
        return s.strip()

    def words_to_visemes(self, w):
        words = w.upper()
        visemes = []
        times = []
        durations = []
        index = 0
        t = 0

        chars = list(words)
        while index < len(chars):
            c = chars[index]
            ruleset = self.rules.get(c)
            if ruleset:
                for rule in ruleset:
                    test = words[:index] + c.lower() + words[index + 1 :]
                    matches = rule["regex"].search(test)
                    if matches:
                        for viseme in rule["visemes"]:
                            if visemes and visemes[-1] == viseme:
                                d = 0.7 * (self.viseme_durations.get(viseme, 1))
                                durations[-1] += d
                                t += d
                            else:
                                d = self.viseme_durations.get(viseme, 1)
                                visemes.append(viseme)
                                times.append(t)
                                durations.append(d)
                                t += d
                        index += rule["move"]
                        break
            else:
                if c == ",":
                    visemes.append("sil")
                    times.append(t)
                    durations.append(self.special_durations.get(c, 0))
                index += 1
                t += self.special_durations.get(c, 0)

        return {
            "words": words,
            "visemes": visemes,
            "times": times,
            "durations": durations,
        }

    def viseme_segmenter(self, w) -> Dict[List, Tuple[str, int, int, int]]:
        """Get the text chunks that mapp to visiemes.
        @returns:
            Dict: words : List of words passed in.
                : viseme_text_mapping: Tuple[str,int,int,int]: (viseme name, start char, end_char, number of visemes in phenome)
        """
        words = w.upper()
        viseme_text_mapping = []
        index = 0
        t = 0

        chars = list(words)
        while index < len(chars):
            c = chars[index]
            ruleset = self.rules.get(c)
            if ruleset:
                for rule in ruleset:
                    test = words[:index] + c.lower() + words[index + 1 :]
                    matches = rule["regex"].search(test)
                    if matches:
                        for viseme in rule["visemes"]:
                            viseme_text_mapping.append(
                                (
                                    viseme,
                                    index,
                                    index + rule["move"],
                                    matches[0],
                                    len(rule["visemes"]),
                                )
                            )
                        index += rule["move"]
                        break
            else:
                index += 1  # else its a non-utterable character to be skipped.
        return {
            "words": words,
            "viseme_mapping": viseme_text_mapping,
        }

    def convert_to_azure_vs(
        self,
        visemes: dict[str, list],
        audio_duration: float,
        begin_sil_duration: float = 0.05,
    ):
        """
        Viseme data here is in normalised time units and needs to be scaled to the actual duration of the audio and conversted to Azure visemes
        """
        normalized_duration = visemes["times"][-1] + visemes["durations"][-1]
        visemes["times"].append(visemes["times"][-1] + visemes["durations"][-1])
        times_in_seconds = [
            t / normalized_duration * audio_duration + begin_sil_duration
            for t in visemes["times"][:-1]
        ]
        start_times = times_in_seconds[:-1]
        end_times = times_in_seconds[1:]
        azure_visemes = [self.azure_vs_id(vs) for vs in visemes["visemes"]]

        vs_az_format = [
            {"start": s, "end": t, "value": v}
            for s, t, v in zip(start_times, end_times, azure_visemes)
        ]
        ret = [{"start": 0.0, "end": str(begin_sil_duration), "id": 0}]
        ret.extend(vs_az_format)
        ret.append({"start": end_times[-1], "end": 1000, "id": 0})
        return ret

    def azure_vs_id(self, viseme: str):
        """Convert the occulus viseme string to the Azure vs id.
        Use for clients that expect Azure viseme IDs and not the string representations.
        """
        try:
            return self.azure_to_occulus_viseme_map.index(viseme)
        except:
            return 0


if __name__ == "__main__":
    converter: LipsyncEn = LipsyncEn()
    print(
        converter.words_to_visemes(
            "Hello, how are you today. Its a lovely day to be out and about isnt it. Hello, how are you today. Its a lovely day to be out and about isnt it. Hello, how are you today. Its a lovely day to be out and about isnt it. Hello, how are you today. Its a lovely day to be out and about isnt it. Hello, how are you today. Its a lovely day to be out and about isnt it."
        )
    )
