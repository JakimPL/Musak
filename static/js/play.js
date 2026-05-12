let _current = null;

export function playSound(path) {
    if (_current) {
        _current.pause();
        _current.currentTime = 0;
    }
    _current = new Audio(path);
    _current.play();
}

export function playAgain(path) {
    if (typeof path !== 'undefined') {
        playSound(path);
    }
}

export function stopSound() {
    if (_current) {
        _current.pause();
        _current.currentTime = 0;
        _current = null;
    }
}