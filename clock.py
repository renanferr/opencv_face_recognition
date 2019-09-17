class Clock:
    def __init__(self, ticks):
        self.ticks = int(ticks)
        self.countdown = self.ticks
        self.done = False

    def tick(self):
        if self.countdown > 0:
            self.countdown -= 1
        else:
            self.done = True

    def is_done(self):
        return self.done

    def restart(self):
        self.countdown = self.ticks
        self.done = False