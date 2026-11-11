"""Console display adapter usable on a PC or CanMV REPL."""
class ConsoleDisplay:
    def __init__(self, output=None):
        self.output = output or print
        self.task = None
        self.counters = None

    def show_task(self, task):
        payload = task.get("payload") if isinstance(task, dict) else getattr(task, "payload", task)
        self.task = payload
        self.output("TASK " + str(payload))

    def show_stats(self, counters):
        counters = counters.as_dict() if hasattr(counters, "as_dict") else dict(counters)
        picks, placements = counters.get("picks"), counters.get("placements")
        if type(picks) is not int or type(placements) is not int or picks < 0 or placements < 0:
            raise ValueError("confirmed nonnegative counters required")
        self.counters = dict(counters)
        self.output("STATS picks=%d placements=%d" % (picks, placements))
