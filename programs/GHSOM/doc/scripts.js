/* $Header: /var/lib/cvs/somtoolbox.bak/doc/scripts.js,v 1.3 2009-11-04 13:42:13 mayer Exp $ */

/* Takes SVN $_Id tag and breaks it down to basic parts */
function getPageInfo() {
    // trim it
    svnId = svnId.substring(svnId.indexOf(": ") + 2, svnId.lastIndexOf(" $"));
    // split it at spaces
    var svnIds = svnId.split(" ");
    // take needed parts
    var filename = svnIds[0].substring(0, svnIds[0].indexOf(","));
    var revision = svnIds[1];
    var date = svnIds[2].replace("/", "-").replace("/", "-");
    var time = svnIds[3];
    var author = svnIds[4];
    if (author == 'mayer') {
        author = 'Rudolf Mayer';
    } else if (author == 'lidy') {
        author = 'Thomas Lidy';
    } else if (author == 'frank') {
        author = 'Jakob Frank';
    }
    // return(''+filename+', revision '+revision+', last modified on '+date+' at '+time+' by '+author);
    return ('Last updated on ' + date /* + ' by ' + author */);
}

function fileIO(U, V) { // read/write stuff from a file...
    var X = !window.XMLHttpRequest ? new ActiveXObject('Microsoft.XMLHTTP') : new XMLHttpRequest();
    X.open(V ? 'PUT' : 'GET', U, false);
    X.setRequestHeader('Content-Type', 'text/html')
    X.send(V ? V : '');
    return X.responseText;
}

// create a table-of-contents of all H1, H2 and H3 elements
function createTOC() {
    tocDiv = document.getElementById('tableOfContents');
    // if there's not table of contents element, we do nothing
    if (tocDiv == null) {
        return;
    }

    // find the nodes to be added to the Page TOC
    var tocTargets = new Array();
    mainNode = document.getElementById('content');
    allNodes = mainNode.getElementsByTagName('*');
    for ( var i = 0; i < allNodes.length; i++) {
        node = allNodes[i];
        if (allNodes[i].nodeName == "H2" || allNodes[i].nodeName == "H3" || allNodes[i].nodeName == "H4") {
            tocTargets.push(allNodes[i]);
        }
    }

    // Remove toc if none or one heading
    if (tocTargets.length <= 1) {
        tocDiv.parentNode.removeChild(tocDiv);
        return;
    }

    // Add the toc contents
    tocDiv.style.display="block";
    tocDiv.innerHTML = "<div class=\"tocHeader\">Table of Contents</div>";
    tocList = document.createElement('ul');
    tocList.className = "tocList";
    tocDiv.appendChild(tocList);

    // Insert elements into our table of contents
    for ( var i = 0; i < tocTargets.length; i++) {
        tocTarget = tocTargets[i];
        if (tocTarget.id == '') {
            tocTarget.id = 'section' + i;
        }
        newItem = document.createElement('li');
        newItem.className = "tocList tocLevel" + tocTarget.nodeName;
        newLink = document.createElement('a');
        newLink.href = '#' + tocTarget.id;
        newLink.innerHTML = tocTarget.innerHTML;
        newItem.appendChild(newLink);
        tocList.appendChild(newItem);
    }
}
