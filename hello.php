<?php
require_once("TensorFlow.php");

$tf = new TensorFlow();
printf("Hello from TensorFlow C library version %s\n", $tf->version());

$sess = $tf->session();

$hello = $tf->constant('Hello, TensorFlow!');
$ret = $sess->run($hello);
var_dump($ret->value());

$hello = $tf->constant(42);
$ret = $sess->run($hello);
var_dump($ret->value());

$hello = $tf->constant([10, 11, 12]);
$ret = $sess->run($hello);
var_dump($ret->value());

$hello = $tf->constant(["a", "b", "c3"]);
$ret = $sess->run($hello);
var_dump($ret->value());


$hello = $tf->constant([[[ 1, 2, 3, 4],[ 5, 6, 7, 8]],
                        [[ 9,10,11,12],[13,14,15,16]],
                        [[17,18,19,20],[21,22,23,24]]]);
$ret = $sess->run($hello);
var_dump($ret->value());

$x = $tf->constant(42);
$y = $tf->constant(5);
$z = $tf->add($x, $y);
$ret = $sess->run($z);
var_dump($ret->value());

$ret = $sess->run(
	$tf->add(
		$tf->constant([42, 17]),
		$tf->constant([5, 25])));
var_dump($ret->value());

$ret = $sess->run(
	$tf->add(
		$tf->placeholder("x", \TF\DOUBLE),
		$tf->placeholder("y", \TF\DOUBLE)),
		["x" => $tf->tensor(42.0), "y" => $tf->tensor(5.0)]);
var_dump($ret->value());
