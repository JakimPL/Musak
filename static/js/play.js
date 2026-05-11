export function playSound(path) {
    const audio = new Audio(path);
    audio.play();
}

export function playAgain(path) {
    if (typeof path !== 'undefined') {
        playSound(path);
    }
}