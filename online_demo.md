---
layout: default
---

<!-- Javascript demo -->
<link rel="stylesheet" href="//code.jquery.com/ui/1.11.0/themes/smoothness/jquery-ui.css">
<script src="//code.jquery.com/jquery-1.10.2.js"></script>
<script src="//code.jquery.com/ui/1.11.0/jquery-ui.js"></script>
<script src="http://www.numericjs.com/lib/numeric-1.2.6.min.js"></script>
<script src="javascripts/morpher.js" type="text/javascript"></script>

<script language="JavaScript">

morpher = new Morpher();

function refreshImage() {
    if (morpher.ready()) {
        var sample = morpher.generate_face();
        image = document.getElementById('face');
        image.src = numeric.imageURL(numeric.mul([sample, sample, sample], 255));
    }
}

function setZ() {
    if (morpher.ready()) {
        for(index in index_mapping) {
            var s_i = $("#slider_" + index).slider("value") / 50.0 - 1.0;
            morpher.set_coordinate(index, s_i);
        }
    }
}

function slidersChanged() {
    setZ();
    refreshImage();
}

function load (filename, length, M, N, callback) {
    var array = [];
    var oReq = new XMLHttpRequest();
    oReq.open("GET", filename, true);
    oReq.responseType = "arraybuffer";

    oReq.onload = function (oEvent) {
        var arrayBuffer = oReq.response; // Note: not oReq.responseText
        if (arrayBuffer) {
            var float32Array = new Float32Array(arrayBuffer);
            for (var i=0; i<M; i++) {
                row = [];
                for(var j=0; j<N; j++) {
                    row.push(float32Array[(N * i) + j]);
                }
                array.push(row);
            }
            console.log('Done loading ' + filename);
            callback();
            refreshImage();
        }
    };

    oReq.send(null);

    return array;
};

$(function() {
    Z = morpher.get_Z();
    for(index in index_mapping) {
        var line = (index < 29 / 2)? 1 : 2;
        name = "slider_" + index;
        $("#labels" + line).append("<td><div>" + index + "</div></td>");
        $("#sliders" + line).append("<td><div id='" + name + "'></div></td>");
        $("#" + name).slider({
            orientation: "vertical",
            range: "min",
            max: 100,
            value: (Z[0][index_mapping[index]] + 1) * 50,
            slide: slidersChanged,
        });
    }
});
$(function() {
   $( "#randombutton" )
   .button({label: "Random"})
     .click(function( event ) {
       if(morpher.ready()) {
           morpher.shuffle();
           Z = morpher.get_Z();
           for(index in index_mapping) {
               $("#slider_" + index).slider("value", (Z[0][index_mapping[index]] + 1) * 50);
           }
           refreshImage();
        }
     });
 });
</script>

# Online Demo

__The online demo needs to load around 35 MB's worth of model parameters,
which might take some time depending on your connection speed.__

<p><img id="face" src="images/loader.gif" alt="face" width="120" /></p>

<center>

<p><div id="randombutton"></div></p>

<p><table style="width:300px">
<tr id="labels1"></tr>
<tr id="sliders1"></tr>
<tr id="labels2"></tr>
<tr id="sliders2"></tr>
</table></p>

</center>
