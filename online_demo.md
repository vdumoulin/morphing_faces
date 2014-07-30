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

$(function() {
    for(index in index_mapping) {
        var line = (index < 29 / 2)? 1 : 2;
        name = "slider_" + index;
        $("#labels" + line).append("<td><div>" + index + "</div></td>");
        $("#sliders" + line).append("<td><div id='" + name + "'></div></td>");
        $("#" + name).slider({
            orientation: "vertical",
            range: "min",
            max: 100,
            value: 50,
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

<img id="face" src="images/loader.gif" alt="face" width="200" />

<div id="randombutton"></div>

<table style="width:300px">
<tr id="labels1"></tr>
<tr id="sliders1"></tr>
<tr id="labels2"></tr>
<tr id="sliders2"></tr>
</table>
