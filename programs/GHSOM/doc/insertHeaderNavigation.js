if (document.getElementById('topbar') == null) {   // load the menu via JS, if it is not present, i.e. if SSI did not work
    document.write(fileIO('headerNavigation.shtml'));
}
