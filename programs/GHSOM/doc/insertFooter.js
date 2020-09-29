if (document.getElementById('footer') == null) {   // load the footer via JS, if it is not present, i.e. if SSI did not work
    document.write(fileIO('footer.shtml'));
}
