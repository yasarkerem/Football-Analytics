COMPETITION_FILES = {
    "England":               "events_England.json",
    "Italy":                 "events_Italy.json",
    "Spain":                 "events_Spain.json",
    "France":                "events_France.json",
    "Germany":               "events_Germany.json",
    "European_Championship": "events_European_Championship.json",
    "World_Cup":             "events_World_Cup.json",
}

MATCH_FILES = {
    "England":               "matches_England.json",
    "Italy":                 "matches_Italy.json",
    "Spain":                 "matches_Spain.json",
    "France":                "matches_France.json",
    "Germany":               "matches_Germany.json",
    "European_Championship": "matches_European_Championship.json",
    "World_Cup":             "matches_World_Cup.json",
}

# Wyscout event type IDs
EVENT_PASS        = 8
EVENT_SHOT        = 10
EVENT_DUEL        = 1
EVENT_FOUL        = 2
EVENT_FREE_KICK   = 3
EVENT_SAVE        = 9
EVENT_OFFSIDE     = 6
EVENT_INTERRUPTION = 5
EVENT_OTHER       = 7

# Pass sub-event IDs
PASS_CROSS        = 80
PASS_HAND         = 81
PASS_HEAD         = 82
PASS_HIGH         = 83
PASS_LAUNCH       = 84
PASS_SIMPLE       = 85
PASS_SMART        = 86

# Tag IDs
TAG_ACCURATE      = 1801
TAG_NOT_ACCURATE  = 1802
TAG_GOAL          = 101
TAG_ASSIST        = 301
TAG_KEY_PASS      = 302
TAG_COUNTER       = 1901
TAG_DANGEROUS     = 2001

# Pitch dimensions (Wyscout normalised coordinates: 0–100 x 0–100)
PITCH_LENGTH = 100.0
PITCH_WIDTH  = 100.0

# Zone boundaries (thirds along pitch length)
DEFENSIVE_THIRD_END  = 33.3
MIDDLE_THIRD_END     = 66.6
