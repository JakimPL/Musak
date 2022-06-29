function playSound(uuid, filename) {
    var path = '../temp/' + uuid + '/' + filename
    var audio = new Audio(path);
    audio.play();
}

function playAgain(directory, audioSource) {
    if (typeof directory !== 'undefined') {
        playSound(directory, audioSource);
    }
}