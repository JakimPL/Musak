class Score {
    constructor() {
        this.reset();
    }

    reset() {
        this.points = 0;
        this.total = 0;
        this.lock = false;
    }

    unlock() {
        this.lock = false;
    }

    update(point) {
        if (!this.lock) {
            this.total += 1;
            if (point) {
                this.points += 1;
            }

            this.lock = true;
        }
    }
}