<?php
require_once("TensorFlow.php");

$tf = new TensorFlow();
$sess = $tf->session();
var_dump($sess->devices());
