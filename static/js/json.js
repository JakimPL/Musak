function openJSON(uuid, filename, func) {
    var path = '../temp/' + uuid + '/' + filename
    $.getJSON(path, function(data) {
        func(data);
    }).fail(function(){
        console.log("An error has occurred during loading a chord info.");
    });
}