function openJSON(path, func) {
    $.getJSON(path, function(data) {
        func(data);
    }).fail(function(){
        console.log("An error has occurred during loading a chord info.");
    });
}