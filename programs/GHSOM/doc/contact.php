<?php
require_once('recaptchalib.php');

$strEmpfaenger = 'somtoolbox@ifs.tuwien.ac.at';
$strFrom       = '"ContactForm" <somtoolbox@ifs.tuwien.ac.at>';

$privatekey = '6LejlQwAAAAAACWQLSN1Rd_EmKueVI5JM-vl30ue';

if (strpos($_SERVER['HTTP_REFERER'], '?') > 0) {
	$http_referer = substr($_SERVER['HTTP_REFERER'], 0, strpos($_SERVER['HTTP_REFERER'], '?'));
} else {
	$http_referer = $_SERVER['HTTP_REFERER'];
}

// Create the mail...
if ($_POST && $_POST['recaptcha_response_field']) {
	
	$resp = recaptcha_check_answer ($privatekey,
		$_SERVER["REMOTE_ADDR"],
		$_POST["recaptcha_challenge_field"],
		$_POST["recaptcha_response_field"]);

	if ($resp->is_valid) {
		$strMailtext = "A message from the SOMToolbox Contact Form\n";
		$strMailtext = $strMailtext . "From: " . trim($_POST['sender']) . "\n";
		$strMailtext = $strMailtext . "Email: " . trim($_POST['sender_mail']) . "\n\n";
		$strMailtext = $strMailtext . "Subject: " . trim($_POST['subject']) . "\n";
		$strMailtext = $strMailtext . "Message:\n" . trim($_POST['message']) . "\n\n\n";

		$strMailtext = $strMailtext . "-- \n";
		$strMailtext = $strMailtext . "Sender:\n";
		$strMailtext = $strMailtext . "Client:  " . $_SERVER['REMOTE_ADDR'] . "\n";
		$strMailtext = $strMailtext . "X-Proxy: " . $_SERVER['HTTP_X_FORWARDED_FOR'] . "\n";
		$strMailtext = $strMailtext . "UserAgent: " . $_SERVER['HTTP_USER_AGENT'] . "\n";

		$strSender = trim($_POST['sender']) . " <" . trim($_POST['sender_mail']) . ">";
		$strSubject = "[SOMToolbox CF] " . trim($_POST['subject']);
		$strFromHeader = "From: SOMToolbox Contact Form on behalf of " . $strSender;
	
		mail($strEmpfaenger, $strSubject, $strMailtext, $strFromHeader . "\nReply-To: " . $strSender)
			 or die("Internal Server Error.");
	}
} else {
	header("Location: $http_referer");
}
?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
  <title>The Java SOMToolbox @ IFS, TU Vienna</title>
  <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1"/>
  <meta http-equiv="Content-Style-Type" content="text/css"/>  
  <link href="style.css" rel="stylesheet" type="text/css"/>
  <link rel="icon" href="favicon.ico" /> 
  <script type="text/javascript" src="scripts.js"></script>
</head>

<body>

<!--#include virtual="headerNavigation.shtml" -->
<script type="text/javascript" src="insertHeaderNavigation.js"></script>

<div class="content" id="content">
<?php if ($resp->is_valid) { ?>
  <h1>Thank you</h1>

  <div class="maintab">
	<p>Thank you for your message.</p>
  </div>
<?php } else { ?>
  <h1>Sending failed.</h1>

  <div class="maintab">
	<p>Could not verify the captcha. Please try again...</p>
	<p>Back to <?php echo "<a href=\"$http_referer\">$http_referer</a>"?></p>
  </div>
<?php } ?> 

  <script type="text/javascript"> var svnId = "$Id: contact.php 3991 2011-01-11 15:50:03Z frank $"; </script>
  <!--#include virtual="footer.shtml" -->
  <script type="text/javascript" src="insertFooter.js"></script>    

</div>

</body>
</html>
