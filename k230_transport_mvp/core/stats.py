class Stats:
    def __init__(self):
        self.picks = 0
        self.placements = 0

    def confirm(self, action):
        if action == "PICK":
            self.picks += 1
        elif action == "PLACE":
            self.placements += 1

    def as_dict(self):
        return {"picks": self.picks, "placements": self.placements}
