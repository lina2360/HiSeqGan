<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
  <title>The Java SOMToolbox @ IFS, TU Vienna - Archive</title>
  <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1"/>
  <meta http-equiv="Content-Style-Type" content="text/css"/>  
  <link href="style.css" rel="stylesheet" type="text/css"/>
  <link rel="icon" href="favicon.ico" /> 
  <script type="text/javascript" src="scripts.js"></script>
</head>

<style>
<!--
h4 {
  margin-bottom: 3px;
}

div.intended {
  margin-left: 20px;
}
-->
</style>

<body>

<!--#include virtual="headerNavigation.shtml" -->
<script type="text/javascript" src="insertHeaderNavigation.js"></script>

<div class="content" id="content">
  <h1>Releases</h1>
  <div class="maintab">
    <p>The Java SOMToolbox is developed and maintained at the <a href="http://www.ifs.tuwien.ac.at/">Institute of Software Technology and
    Interactive System</a> at the <a href="http://www.tuwien.ac.at">Vienna University of Technology</a>, as a <strong>research prototype</strong>.</p>
	<p>This page lists the managed release packages. While older versions are also available, we strongly recomend to <strong>use the most recent release</strong>.</p>
	<p></p>
    <p>The Java SOMToolbox is licensed under the <a href="license.html">Apache License, Version 2.0.</a>, and you are free to use
    the software for any kind of purpose that conforms with the license.</p>
  <h2>Available Packages</h2>
  	<p>We provide simple packages as well as installers for various plattforms and a <a href="#debian">debian repository</a>.</p>
  <?php 
	// from http://at2.php.net/manual/en/function.filesize.php#99333
	function format_size($size) {
    	$sizes = array(" Bytes", " KB", " MB", " GB", " TB", " PB", " EB", " ZB", " YB");
		if ($size == 0) { return(''); } else {
			return (round($size/pow(1024, ($i = floor(log($size, 1024)))), $i > 1 ? 0 : 0) . $sizes[$i]); 
		}
	}

	$archive = dirname(__FILE__) . "/../../download";
	$archive_link = "../download";
	$nicename['somtoolbox'] = 'Binary distribution package';
	$nicename['somtoolbox+src'] = 'Binary and source distribution package';
	$nicename['somtoolbox+opt'] = false;
	$nicename['somtoolbox+opt+src'] = false;
	$nicename['SOMToolbox-setup'] = 'Offline Installer';
	$nicename['SOMToolbox-netsetup'] = 'Net-Installer';
			
	foreach (scandir($archive, 1) as $file) {
		if (preg_match('/^somtoolbox(\+.*)?_/i', $file)) {
			if (preg_match('/\.pack-/', $file)) continue;

			$dot = strrpos($file, '.', -4);
			$ftype = substr($file, $dot + 1);
			$basename = substr($file, 0, $dot);
			$group = preg_replace('/_.*[0-9]([+-]?)/', '$1', $basename);
			//$version = preg_filter(array('/somtoolbox.*_([0-9])/i', '/[+-][^0-9]+$/i'),array('$1',''), $basename);
			$version = preg_replace('/somtoolbox.*_([0-9])/i','$1', $basename);
			$version = preg_replace('/[+-][^0-9]+$/i','', $version);

			if ($version && $nicename[$group] && $ftype) {
				$entry[$version][$group][$ftype] = $file;
			}
		}
	}
  
	if (is_array($entry)) {
		$versions = array_keys($entry);
		// Put new versions first...
		rsort($versions);
		foreach ($versions as $version) {
			$major = preg_replace('/^([0-9\.]+-[0-9]+)\.(.*)$/','$1',$version);
			$svn =   preg_replace('/^([0-9\.]+-[0-9]+)\.(.*)$/','$2',$version);
                        printf("<div style=\"margin-top: 1em;\"><h3 id=\"%s\" style=\"display: inline; margin-top 1em;\">%s</h3> <span style=\"font-size: 
90%%\">%s</span></div>\n<div class=\"intended\" 
style=\"padding-top: 0.5em\">\n", $version, $major, $svn);
			foreach (array_keys($entry[$version]) as $group) {
				if ($nicename[$group]) {
					printf("    <div style=\"padding: 0 0.5em 0 0.5em\">%s: ", $nicename[$group]);
					foreach (array_keys($entry[$version][$group]) as $ftype) {
						$link = sprintf("%s/%s", $archive_link, $entry[$version][$group][$ftype]);
						$file = sprintf("%s/%s", $archive, $entry[$version][$group][$ftype]);
						printf("<span style=\"font-size: 75%%;\"><a href=\"%s\">%s (ca. %s)</a></span>\n",$link, $ftype, format_size(filesize($file)));
					}
				printf("</div>");
				}
			}
			printf("  </div>\n");
		}
	}
  ?>
  
  <span id="debian"><h3>Debian repository</h3></span>
      <div class="intended">    
      A convenient solution for Debian (and other Debian-based operating systems such as Ubuntu).<br />
      You can add the repository to your system in two ways:
      <ol>
        <li>
        <p>Add the following APT line to your <span>/etc/apt/sources.list</span> file, or to a new file in your <span>/etc/apt/sources.list.d</span>
        directory.</p>
        <pre>deb http://olymp.ifs.tuwien.ac.at/apt <strong>stable</strong> main #IFS/Vienna University of Technology</pre>
        <p>Then, you need to reload the package information, install the repository keyring, and reload the package information
        again:</p>
        <pre>sudo aptitude update
sudo aptitude install olymp-keyring
sudo aptitude update</pre></li>
        <li>
        <p>You can also do all this steps with the following command:</p>
        <pre>sudo wget --output-document=/etc/apt/sources.list.d/ifs-Vienna.list http://www.ifs.tuwien.ac.at/dm/somtoolbox/ifs-Vienna.list &amp;&amp; sudo apt-get --quiet update &amp;&amp; sudo apt-get --yes --quiet --allow-unauthenticated install olymp-keyring &amp;&amp; sudo apt-get --quiet update</pre>
        </li>
      </ol>
      <p>Finally, you should install the somtoolbox packages via aptitude or synaptic.</p>
      <p>There is also a nightly build of the debian package, if you want to always receive the latest developments. However, be
      advised that these nightly builds might include bugs, ...</p>
      <pre>deb http://olymp.ifs.tuwien.ac.at/apt nightly main #IFS/Vienna University of Technology</pre>
    </div>  
  </div>

  <script type="text/javascript"> var svnId = "$Id: download-releases.php 3913 2010-11-04 13:38:17Z frank $"; </script>
  <!--#include virtual="footer.shtml" -->
  <script type="text/javascript" src="insertFooter.js"></script>    

</div>

</body>
</html>
  
