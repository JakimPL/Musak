function playSound(path) {
    var audio = new Audio(path);
    audio.play();
}

function playAgain(path) {
    if (typeof path !== 'undefined') {
        playSound(path);
    }
}