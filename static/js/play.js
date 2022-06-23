function playSound(uuid, filename) {
    var path = '../temp/' + uuid + '/' + filename
    var audio = new Audio(path);
    audio.play();
}