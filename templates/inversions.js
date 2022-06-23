function playChord(uuid) {
    var path = 'temp/' + uuid + '/chord.mp3'
    var audio = new Audio(path);
    audio.play();
}

function ajax(event) {
    event.preventDefault();
    $.ajax({
        type: 'POST',
        url: "{% url 'submit' %}",
        data: {
            submit: $('#inversion_generator').serialize(),
            csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val()
        },
        success: function(response) {
            if ('directory' in response) {
                playChord(response['directory']);
            }
        },
        error: function(response){
            alert('Bad request');
        }
    });
}